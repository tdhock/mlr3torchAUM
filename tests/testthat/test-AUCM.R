if (torch::torch_is_installed() && requireNamespace("mlr3torch")) {
  test_that("Step1: AUCM returns 0", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.5))
    label <- torch::torch_tensor(c(1))
    out <- AUCM(pred, label)
    expect_equal(out$item(), 0, tolerance = 1e-6)
  })

  test_that("Step2: positive_ratio returns the ratio of positives", {
    skip_on_cran()
    expect_equal(positive_ratio(torch::torch_tensor(c(0, 0, 1, 1)))$item(), 0.5, tolerance = 1e-6)
    expect_equal(positive_ratio(torch::torch_tensor(c(0, 1, 1, 1)))$item(), 0.75, tolerance = 1e-6)
  })

  test_that("Step3: AUCM positive squared term (isolated)", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0, 0, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    # a=0: (1-0.5)*(0.35^2 + 0.8^2)/4 = 0.0953125
    expect_equal(AUCM(pred, label)$item(), 0.0953125, tolerance = 1e-6)
    # a=0.2: (1-0.5)*((0.15)^2 + (0.6)^2)/4 = 0.0478125
    expect_equal(AUCM(pred, label, a = 0.2)$item(), 0.0478125, tolerance = 1e-6)
  })

  test_that("Step4: AUCM negative squared term (isolated)", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.2, 0.6, 0, 0))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    # b=0: 0.5*(0.2^2 + 0.6^2)/4 = 0.05
    expect_equal(AUCM(pred, label)$item(), 0.05, tolerance = 1e-6)
    # b=0.1: 0.5*((0.1)^2 + (0.5)^2)/4 = 0.0325
    expect_equal(AUCM(pred, label, b = 0.1)$item(), 0.0325, tolerance = 1e-6)
  })

  test_that("Step5: AUCM full v1 forward matches LibAUC", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    expect_equal(AUCM(pred, label, a = 0, b = 0, alpha = 0, margin = 1)$item(),
      0.1165625,
      tolerance = 1e-6
    )
    expect_equal(AUCM(pred, label, a = 0.3, b = 0.6, alpha = 0.5, margin = 1)$item(),
      0.1740625,
      tolerance = 1e-6
    )
    expect_equal(AUCM(pred, label, a = 0.3, b = 0.6, alpha = 0.5, margin = 2)$item(),
      0.4240625,
      tolerance = 1e-6
    )
  })
  test_that("Step6: nn_AUCM_loss exposes a/b/alpha as trainable parameters", {
    skip_on_cran()
    loss_fn <- nn_AUCM_loss(add_sigmoid = FALSE)
    # a/b/alpha are parameters, with initial value of 0
    expect_equal(loss_fn$a$item(), 0, tolerance = 1e-6)
    expect_equal(loss_fn$b$item(), 0, tolerance = 1e-6)
    expect_equal(loss_fn$alpha$item(), 0, tolerance = 1e-6)
    # margin is just hyperparameter
    expect_equal(length(loss_fn$parameters), 3)
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    expect_equal(loss_fn(pred, label)$item(), 0.1165625, tolerance = 1e-6)
  })

  test_that("Step7: gradients flow; alpha's grad sign exposes the min-max", {
    skip_on_cran()
    loss_fn <- nn_AUCM_loss(add_sigmoid = FALSE) # a=b=alpha=0, margin=1
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    loss <- loss_fn(pred, label)
    expect_true(loss$requires_grad)
    loss$backward()
    expect_equal(as.numeric(loss_fn$a$grad), -0.2875, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$b$grad), -0.125, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$alpha$grad), 0.3375, tolerance = 1e-6)
    expect_lt(as.numeric(loss_fn$a$grad), 0)
    expect_lt(as.numeric(loss_fn$b$grad), 0)
    # alpha > 0 which is problematic!
    expect_gt(as.numeric(loss_fn$alpha$grad), 0)
  })


  test_that("Step9: pesg_alpha_step ascends alpha toward (margin+b-a), clamps >= 0", {
    skip_on_cran()
    loss_fn <- nn_AUCM_loss(margin = 1, add_sigmoid = FALSE)
    torch::with_no_grad({
      # target = margin - (0.575-0.25) = 0.675
      loss_fn$a$fill_(0.575)
      loss_fn$b$fill_(0.25)
      loss_fn$alpha$fill_(0)
    })
    # one step: (lr=0.1)：0 + 0.1*(2*0.675 - 0) = 0.135
    pesg_alpha_step(loss_fn, lr = 0.1)
    expect_equal(loss_fn$alpha$item(), 0.135, tolerance = 1e-6)
    # multi step: converge to target = 0.675
    for (i in 1:200) pesg_alpha_step(loss_fn, lr = 0.1)
    expect_equal(loss_fn$alpha$item(), 0.675, tolerance = 1e-4)
    # when target is negative, clamp to 0，no penalties
    loss_fn2 <- nn_AUCM_loss(margin = 0.1, add_sigmoid = FALSE)
    torch::with_no_grad({
      # target = margin - (1-0) = -0.9
      loss_fn2$a$fill_(1)
      loss_fn2$b$fill_(0)
      loss_fn2$alpha$fill_(0.05)
    })
    pesg_alpha_step(loss_fn2, lr = 1)
    expect_equal(loss_fn2$alpha$item(), 0, tolerance = 1e-6)
  })

  test_that("Step10: pesg_step descends a/b, ascends alpha, zeros grads", {
    skip_on_cran()
    loss_fn <- nn_AUCM_loss(margin = 1, add_sigmoid = FALSE) # a=b=alpha=0
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    loss <- loss_fn(pred, label)
    loss$backward() # a$grad=-0.2875, b$grad=-0.125, alpha$grad=0.3375
    pesg_step(loss_fn, lr = 0.1)
    expect_equal(loss_fn$a$item(), 0.02875, tolerance = 1e-6) # 0 - 0.1*(-0.2875)
    expect_equal(loss_fn$b$item(), 0.0125, tolerance = 1e-6) # 0 - 0.1*(-0.125)
    # 2*0.1*(margin + b - a - alpha) = 2*0.1*(1 + 0.0125 - 0.02875 - 0) = 0.19675
    expect_equal(loss_fn$alpha$item(), 0.19675, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$a$grad), 0, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$b$grad), 0, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$alpha$grad), 0, tolerance = 1e-6)
  })

  test_that("Step12: nn_AUCM_loss + PESG callback trains in mlr3torch", {
    skip_on_cran()
    set.seed(1)
    torch::torch_manual_seed(1)
    n <- 200
    x1 <- rnorm(n)
    x2 <- rnorm(n)
    y <- factor(ifelse(plogis(1.5 * x1 - x2) > 0.7, "pos", "neg"), levels = c("neg", "pos"))
    task <- mlr3::TaskClassif$new("toy", data.frame(x1, x2, y), target = "y", positive = "pos")
    L <- mlr3torch::LearnerTorchMLP$new(task_type = "classif")
    tl <- mlr3torch::as_torch_loss(nn_AUCM_loss)
    tl$param_set$set_values(add_sigmoid = FALSE)
    L$loss <- tl
    L$optimizer <- mlr3torch::t_opt("sgd", lr = 0.05)
    L$callbacks <- make_pesg_callback(lr = 0.05)
    L$predict_type <- "prob"
    L$param_set$set_values(epochs = 30, batch_size = 32, neurons = 8, shuffle = FALSE, seed = 1)
    L$train(task)
    lf <- L$model$loss_fn
    expect_gt(as.numeric(lf$a), as.numeric(lf$b)) # pos avg > neg avg
    expect_gte(as.numeric(lf$alpha), 0) # clamp
    auc <- L$predict(task)$score(mlr3::msr("classif.auc"))
    expect_gt(auc, 0.8) # good auc
  })

  test_that("Step13: adam_step implements Adam, matches torch::optim_adam", {
    skip_on_cran()
    lr <- 0.1
    betas <- c(0.9, 0.999)
    eps <- 1e-8
    grads <- c(1.0, -2.0, 0.5, 3.0)
    p_torch <- torch::torch_tensor(0.0, requires_grad = TRUE) # simulate the param
    opt <- torch::optim_adam(list(p_torch), lr = lr, betas = betas, eps = eps)
    ground_truth <- numeric(length(grads))
    for (i in seq_along(grads)) {
      opt$zero_grad() # clean out grad of the param
      (grads[i] * p_torch)$backward() # grad is just the grad[i]
      opt$step()
      ground_truth[i] <- p_torch$item() # updated param
    }
    p_torch <- torch::torch_tensor(0.0, requires_grad = TRUE)
    state <- NULL
    for (i in seq_along(grads)) {
      if (i > 1) p_torch$grad$zero_()
      (grads[i] * p_torch)$backward()
      state <- adam_step(p_torch, lr = lr, betas = betas, eps = eps, state = state)
      expect_equal(p_torch$item(), ground_truth[i], tolerance = 1e-6)
      if (i == 1) { # check the m and v in state in the first step, grad = 1.0
        expect_equal(as.numeric(state$m), 0.1, tolerance = 1e-6) # m = 0.9*0 + 0.1*1
        expect_equal(as.numeric(state$v), 0.001, tolerance = 1e-6) # v = 0.999*0 + 0.001*1^2
        expect_equal(state$t, 1)
      }
    }
  })

  test_that("Step14: pesg_step/callback mode='adam' uses Adam for a/b (Route A+)", {
    skip_on_cran()
    lf <- nn_AUCM_loss(margin = 1, add_sigmoid = FALSE)
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    lf(pred, label)$backward()
    expect_equal(as.numeric(lf$a$grad), -0.2875, tolerance = 1e-6)
    expect_equal(as.numeric(lf$b$grad), -0.125, tolerance = 1e-6)
    st <- pesg_step(lf, lr = 0.1, mode = "adam", state = NULL)
    expect_equal(lf$a$item(), 0.1, tolerance = 1e-6) # first step: lr*sign(grad) = 0.1
    expect_equal(lf$b$item(), 0.1, tolerance = 1e-6)
    expect_equal(st$a$t, 1)
    expect_equal(st$b$t, 1)
    set.seed(1)
    torch::torch_manual_seed(1)
    n <- 200
    x1 <- rnorm(n)
    x2 <- rnorm(n)
    y <- factor(ifelse(plogis(1.5 * x1 - x2) > 0.7, "pos", "neg"), levels = c("neg", "pos"))
    task <- mlr3::TaskClassif$new("toy", data.frame(x1, x2, y), target = "y", positive = "pos")
    L <- mlr3torch::LearnerTorchMLP$new(task_type = "classif")
    tl <- mlr3torch::as_torch_loss(nn_AUCM_loss)
    tl$param_set$set_values(add_sigmoid = FALSE)
    L$loss <- tl
    L$optimizer <- mlr3torch::t_opt("sgd", lr = 0.05)
    L$callbacks <- make_pesg_callback(lr = 0.05, mode = "adam")
    L$predict_type <- "prob"
    L$param_set$set_values(epochs = 30, batch_size = 32, neurons = 8, shuffle = FALSE, seed = 1)
    L$train(task)
    lf2 <- L$model$loss_fn
    expect_gt(as.numeric(lf2$a), as.numeric(lf2$b)) # pos avg > neg avg
    expect_gte(as.numeric(lf2$alpha), 0) # clamped
    expect_gt(L$predict(task)$score(mlr3::msr("classif.auc")), 0.8) # pretty good
  })

  test_that("Step 15: test clamp grad function", {
    skip_on_cran()
    grad <- torch::torch_tensor(c(2, -3, 0.3, 0.4), dtype = torch::torch_float32())
    grad_clamped_1 <- pesg_clamp_grad(grad, 1)
    expect_equal(as.numeric(grad_clamped_1), c(1, -1, 0.3, 0.4), tolerance = 1e-6)
    grad_clamped_2 <- pesg_clamp_grad(grad, 0.4)
    expect_equal(as.numeric(grad_clamped_2), c(0.4, -0.4, 0.3, 0.4), tolerance = 1e-6)
  })

  test_that("Step 16: test weight decay", {
    skip_on_cran()
    grad1 <- torch::torch_tensor(c(2, -3, 0.3, 0.4), dtype = torch::torch_float32())
    p1 <- torch::torch_tensor(c(5, 5, 5, 5), dtype = torch::torch_float32())
    grad_only_clamp <- pesg_d_p(grad1, p1, clamp_value = 1, weight_decay = 0)
    expect_equal(as.numeric(grad_only_clamp), c(1, -1, 0.3, 0.4), tolerance = 1e-6)
    grad2 <- torch::torch_tensor(c(0.3, 0.4), dtype = torch::torch_float32())
    p2 <- torch::torch_tensor(c(2, 4), dtype = torch::torch_float32())
    grad_only_weight_decay <- pesg_d_p(grad2, p2, clamp_value = 1, weight_decay = 0.5)
    expect_equal(as.numeric(grad_only_weight_decay), c(1.3, 2.4), tolerance = 1e-6)
    grad_both <- pesg_d_p(grad2, p2, clamp_value = 0.3, weight_decay = 0.5)
    expect_equal(as.numeric(grad_both), c(1.3, 2.3), tolerance = 1e-6)
  })

  test_that("Step 17: test epoch decay", {
    skip_on_cran()
    grad <- torch::torch_tensor(c(0.3, 0.4), dtype = torch::torch_float32())
    p <- torch::torch_tensor(2, dtype = torch::torch_float32()) # broadcast
    model_ref <- torch::torch_tensor(1, dtype = torch::torch_float32()) # broadcast
    grad_clamp_epoch_decay <- pesg_d_p(grad, p,
      clamp_value = 0.3, weight_decay = 0,
      epoch_decay = 0.1, model_ref = model_ref
    )
    expect_equal(as.numeric(grad_clamp_epoch_decay),
      c(0.4, 0.4), # 0.3 + 2 * 0 + 0.1 * (2 - 1)
      tolerance = 1e-6
    )
    grad_all <- pesg_d_p(grad, p,
      clamp_value = 0.3, weight_decay = 0.5,
      epoch_decay = 0.1, model_ref = model_ref
    )
    expect_equal(as.numeric(grad_all),
      c(1.4, 1.4), # 0.3 + 2 * 0.5 + 0.1 * (2 - 1)
      tolerance = 1e-6
    )
  })

  test_that("Step 18: test momentum", {
    skip_on_cran()
    dp <- torch::torch_tensor(c(0.3, 0.4), dtype = torch::torch_float32())
    buf <- torch::torch_tensor(c(1, 1), dtype = torch::torch_float32())
    g_no_buf <- pesg_buffer_momentum(dp, buf, momentum = 1)
    expect_equal(as.numeric(g_no_buf), as.numeric(dp), tolerance = 1e-6)
    g_no_dp <- pesg_buffer_momentum(dp, buf, momentum = 0)
    expect_equal(as.numeric(g_no_dp), as.numeric(buf), tolerance = 1e-6)
    g_both <- pesg_buffer_momentum(dp, buf, momentum = 0.5)
    expect_equal(as.numeric(g_both), c(0.65, 0.7), tolerance = 1e-6)
    buf0 <- torch::torch_tensor(0, dtype = torch::torch_float32())
    g_init_buff <- pesg_buffer_momentum(dp, buf0, momentum = 0.5)
    expect_equal(as.numeric(g_init_buff), c(0.15, 0.2), tolerance = 1e-6)
  })

  test_that("Step 19: test primal step", {
    skip_on_cran()
    grad1 <- torch::torch_tensor(c(0.5, -0.5), dtype = torch::torch_float32())
    p1 <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float32())
    buf1 <- torch::torch_tensor(c(0, 0), dtype = torch::torch_float32())
    model_acc1 <- torch::torch_tensor(c(0, 0), dtype = torch::torch_float32())
    model_acc_updated1 <- pesg_primal_step(p1, grad1,
      lr = 0.1, clamp_value = 10, weight_decay = 0,
      epoch_decay = 0, model_ref = 0, momentum = 1, buffer = buf1, model_acc = model_acc1
    )$model_acc
    expect_equal(as.numeric(model_acc_updated1), c(0.95, 2.05), tolerance = 1e-6) # calculate by hand
    grad2 <- torch::torch_tensor(c(2, -2), dtype = torch::torch_float32())
    p2 <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float32())
    buf2 <- torch::torch_tensor(c(1, 1), dtype = torch::torch_float32())
    model_acc2 <- torch::torch_tensor(c(10, 10), dtype = torch::torch_float32())
    model_acc_updated2 <- pesg_primal_step(p2, grad2,
      lr = 0.1, clamp_value = 1, weight_decay = 0.5,
      epoch_decay = 0, model_ref = 0, momentum = 0.5, buffer = buf2, model_acc = model_acc2
    )$model_acc
    expect_equal(as.numeric(model_acc_updated2), c(10.875, 11.95), tolerance = 1e-6) # calculate by hand
  })

  test_that("Step 20: test update regularizer", {
    skip_on_cran()
    model_acc1 <- torch::torch_tensor(c(10, 20), dtype = torch::torch_float32())
    T1 <- 5
    model_ref1 <- pesg_update_regularizer(model_acc1, T1)$model_ref
    expect_equal(as.numeric(model_ref1), c(2, 4), tolerance = 1e-6)
    model_acc2 <- torch::torch_tensor(c(3, 6, 9), dtype = torch::torch_float32())
    T2 <- 3
    model_ref2 <- pesg_update_regularizer(model_acc2, T2)$model_ref
    expect_equal(as.numeric(model_ref2), c(1, 2, 3), tolerance = 1e-6)
  })

  test_that("Step 21: test decay learning rate", {
    skip_on_cran()
    lr1 <- 0.1
    decay_factor1 <- 2
    lr_updated1 <- pesg_update_lr(lr1, decay_factor1)
    expect_equal(lr_updated1, 0.05)
    lr2 <- 0.05
    decay_factor2 <- 10
    lr_updated2 <- pesg_update_lr(lr2, decay_factor2)
    expect_equal(lr_updated2, 0.005)
  })

  test_that("Step 22: optimizer", {
    skip_on_cran()
    x <- torch::torch_tensor(c(1, 2, 3), dtype = torch::torch_float32(), requires_grad = TRUE)
    opt <- optim_pesg(list(x),
      lr = 0.1, clamp_value = 2, weight_decay = 0.8,
      epoch_decay = 0.1, momentum = 0.5, decay_factor = 2
    )
    x$sum()$backward()
    grad <- x$grad$clone() # 1,1,1
    opt$step()
    x_ref <- torch::torch_tensor(c(1, 2, 3), dtype = torch::torch_float32())
    pesg_primal_step(x_ref, grad,
      lr = 0.1, clamp_value = 2, weight_decay = 0.8,
      epoch_decay = 0.1, model_ref = 0, momentum = 0.5, buffer = NULL, model_acc = 0
    ) # same hyperparameters; buffer = NULL = first encounter, matches optim_pesg
    expect_equal(as.numeric(x), as.numeric(x_ref))
    expect_true(!is.null(mlr3torch::as_torch_optimizer(optim_pesg)))
  })

  test_that("Step 23: pesg full step for loss parameters", {
    skip_on_cran()
    loss_fn <- nn_AUCM_loss(margin = 1, add_sigmoid=FALSE)
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8),
      dtype = torch::torch_float32(), requires_grad = TRUE
    )
    target <- torch::torch_tensor(c(0, 0, 1, 1),
      dtype = torch::torch_long()
    )
    loss_fn(pred, target)$backward()
    state_a <- list(buffer = 0, model_acc = 0, model_ref = 0, T = 0)
    state_b <- list(buffer = 0, model_acc = 0, model_ref = 0, T = 0)
    pesg_full_step(loss_fn,
      lr = 0.1, clamp_value = 10, weight_decay = 0,
      epoch_decay = 0, momentum = 1, state_a = state_a, state_b = state_b
    )
    # Same as Step 10
    expect_equal(loss_fn$a$item(), 0.02875, tolerance = 1e-6) # 0 - 0.1*(-0.2875)
    expect_equal(loss_fn$b$item(), 0.0125, tolerance = 1e-6) # 0 - 0.1*(-0.125)
    # 2*0.1*(margin + b - a - alpha) = 2*0.1*(1 + 0.0125 - 0.02875 - 0) = 0.19675
    expect_equal(loss_fn$alpha$item(), 0.19675, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$a$grad), 0, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$b$grad), 0, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$alpha$grad), 0, tolerance = 1e-6)
  })

  test_that("Step 24: full callback constructs", {
    skip_on_cran()
    cb <- make_pesg_callback_full(
      lr = 0.1, clamp_value = 1, weight_decay = 0,
      epoch_decay = 0, momentum = 0.9, decay_factor = 2
    )
    expect_true(inherits(cb, "TorchCallback"))
  })

  test_that("Step 25: e2e optim_pesg + full PESG callback", {
    skip_on_cran()
    set.seed(1)
    torch::torch_manual_seed(1)
    n <- 200
    x1 <- rnorm(n)
    x2 <- rnorm(n)
    y <- factor(ifelse(plogis(1.5 * x1 - x2) > 0.6, "pos", "neg"), levels = c("pos", "neg"))
    task <- mlr3::TaskClassif$new("t", data.frame(x1, x2, y), target = "y", positive = "pos")
    L <- mlr3torch::LearnerTorchMLP$new(task_type = "classif")
    tl <- mlr3torch::as_torch_loss(nn_AUCM_loss)
    tl$param_set$set_values(add_sigmoid = FALSE)
    L$loss <- tl
    opt <- mlr3torch::as_torch_optimizer(optim_pesg)
    opt$param_set$set_values(
      lr = 0.1, clamp_value = 1, weight_decay = 1e-4,
      epoch_decay = 1e-3, momentum = 0.9, decay_factor = 2
    )
    L$optimizer <- opt
    L$callbacks <- make_pesg_callback_full(
      lr = 0.1, clamp_value = 1, weight_decay = 1e-4,
      epoch_decay = 1e-3, momentum = 0.9, decay_factor = 2
    )
    L$predict_type <- "prob"
    L$param_set$set_values(epochs = 10, batch_size = 32, neurons = 4, shuffle = TRUE, seed = 1)
    L$train(task)
    loss_fn <- L$model$loss_fn
    expect_gt(L$predict(task)$score(mlr3::msr("classif.auc")), 0.8)
    expect_gt(as.numeric(loss_fn$a), as.numeric(loss_fn$b))
    expect_gte(as.numeric(loss_fn$alpha), 0)
  })

  test_that("momentum = 0 falls back to plain SGD", {
    skip_on_cran()
    # Gold:
    # #  uv run --with 'libauc==1.4.0' --with torch --with 'numpy<2' --python 3.11 python test.py
    # import torch
    # from libauc.optimizers import PESG
    # from libauc.losses import AUCMLoss
    # G = [(1., 2.), (2., 4.), (3., 6.), (4., 8.)]
    # loss_fn = AUCMLoss(margin=1.0, device='cpu')
    # p = torch.nn.Parameter(torch.tensor([1., 2.]))
    # opt = PESG([p], loss_fn=loss_fn, lr=0.1, momentum=0.0, clip_value=10.,
    #         weight_decay=0., epoch_decay=0., verbose=False,
    #         device='cpu')
    # for t in range(4):
    #     p.grad = torch.tensor(list(G[t]))
    #     opt.step()
    #     st = opt.state[p]
    #     print('   ', ' '.join('%.10g' % v for v in p.data.tolist()))
    # # output:
    #     # 0.8999999762 1.799999952
    #     # 0.6999999881 1.399999976
    #     # 0.3999999762 0.7999999523
    #     # -2.980232239e-08 -5.960464478e-08
    g1 <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float32())
    g2 <- torch::torch_tensor(c(2, 4), dtype = torch::torch_float32())
    g3 <- torch::torch_tensor(c(3, 6), dtype = torch::torch_float32())
    g4 <- torch::torch_tensor(c(4, 8), dtype = torch::torch_float32())
    p <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float32())
    res <- pesg_primal_step(p, g1,
      lr = 0.1, clamp_value = 10, weight_decay = 0,
      epoch_decay = 0, model_ref = 0, momentum = 0, buffer = NULL, model_acc = 0
    )
    m0_t1 <- as.numeric(p)
    res <- pesg_primal_step(p, g2,
      lr = 0.1, clamp_value = 10, weight_decay = 0,
      epoch_decay = 0, model_ref = 0, momentum = 0, buffer = res$buffer, model_acc = 0
    )
    m0_t2 <- as.numeric(p)
    res <- pesg_primal_step(p, g3,
      lr = 0.1, clamp_value = 10, weight_decay = 0,
      epoch_decay = 0, model_ref = 0, momentum = 0, buffer = res$buffer, model_acc = 0
    )
    m0_t3 <- as.numeric(p)
    res <- pesg_primal_step(p, g4,
      lr = 0.1, clamp_value = 10, weight_decay = 0,
      epoch_decay = 0, model_ref = 0, momentum = 0, buffer = res$buffer, model_acc = 0
    )
    m0_t4 <- as.numeric(p)
    expect_equal(m0_t1, c(0.8999999762, 1.799999952), tolerance = 1e-7)
    expect_equal(m0_t2, c(0.6999999881, 1.399999976), tolerance = 1e-7)
    expect_equal(m0_t3, c(0.3999999762, 0.7999999523), tolerance = 1e-7)
    expect_equal(m0_t4, c(-2.980232239e-08, -5.960464478e-08), tolerance = 1e-7)
  })

  test_that("test class_mean", {
    skip_on_cran()
    x_with_zero <- torch::torch_tensor(c(2, 0, 4), dtype = torch::torch_float32())
    mask_first_two <- torch::torch_tensor(c(1, 1, 0), dtype = torch::torch_float32())
    expect_equal(as.numeric(class_mean(x_with_zero, mask_first_two)), 1, tolerance = 1e-6)
    x_all_selected <- torch::torch_tensor(c(0.5, 1.5), dtype = torch::torch_float32())
    mask_all <- torch::torch_tensor(c(1, 1), dtype = torch::torch_float32())
    expect_equal(as.numeric(class_mean(x_all_selected, mask_all)), 1, tolerance = 1e-6)
    x_unused <- torch::torch_tensor(c(9, 9), dtype = torch::torch_float32())
    mask_empty <- torch::torch_tensor(c(0, 0), dtype = torch::torch_float32())
    expect_true(is.nan(as.numeric(class_mean(x_unused, mask_empty)))) # divided by 0
    x_four <- torch::torch_tensor(c(1, 2, 3, 4), dtype = torch::torch_float32())
    mask_last_two <- torch::torch_tensor(c(0, 0, 1, 1), dtype = torch::torch_float32())
    expect_equal(as.numeric(class_mean(x_four, mask_last_two)), 3.5, tolerance = 1e-6)
  })

  test_that("test AUCM version = 'v2'", {
    skip_on_cran()
    # Goldens from LibAUC 2.0.1, via:
    #   uv run --with 'libauc==2.0.1' --with torch --with 'numpy<2' --python 3.11 python <script_name>
    #   import torch
    #   from libauc.losses.auc import AUCMLoss
    #   s = torch.tensor([0.1, 0.4, 0.35, 0.8]).view(-1, 1)
    #   y = torch.tensor([0., 0., 1., 1.]).view(-1, 1)
    #   def golden(a, b, al, m=1.0, version='v1'):
    #       f = AUCMLoss(margin=m, version=version)
    #       with torch.no_grad():
    #           f.a.fill_(a); f.b.fill_(b); f.alpha.fill_(al)
    #       out = float(f(s, y))
    #       return out / (0.5 * 0.5) if version == 'v1' else out
    #   golden(0, 0, 0)                        # 0.4662500321865082
    #   golden(0.3, 0.6, 0.5)                  # 0.6962500214576721
    #   float(AUCMLoss(margin=1.0, version='v1')(s, y))   # 0.11656250804662704
    #   golden(0.3, 0.6, 0.5, version='v2')    # 0.5712500214576721, wrong, bug!
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    expect_equal(
      AUCM(pred, label, a = 0, b = 0, alpha = 0, margin = 1, version = "v2")$item(),
      0.4662500,
      tolerance = 1e-6
    )
    expect_equal(
      AUCM(pred, label, a = 0.3, b = 0.6, alpha = 0.5, margin = 1, version = "v2")$item(),
      0.6962500,
      tolerance = 1e-6
    )
    expect_equal(AUCM(pred, label, a = 0, b = 0, alpha = 0, margin = 1)$item(),
      0.1165625,
      tolerance = 1e-6
    )
  })

  test_that("test v2 equals v1 divided by p*(1-p)", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    for (label_values in list(c(0, 0, 1, 1), c(0, 0, 0, 1), c(0, 1, 1, 1))) {
      label <- torch::torch_tensor(label_values)
      p <- as.numeric(positive_ratio(label))
      for (par in list(c(0, 0, 0, 1), c(0.3, 0.6, 0.5, 1), c(0.2, 0.1, 0.7, 2))) {
        v1 <- AUCM(pred, label, par[1], par[2], par[3], par[4])$item()
        v2 <- AUCM(pred, label, par[1], par[2], par[3], par[4], version = "v2")$item()
        expect_equal(v2, v1 / (p * (1 - p)), tolerance = 1e-6)
      }
    }
  })

  test_that("test v2 does not reproduce the LibAUC cross-term flaw", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    v2 <- AUCM(pred, label, a = 0.3, b = 0.6, alpha = 0.5, margin = 1, version = "v2")$item()
    expect_false(isTRUE(all.equal(v2, 0.5712500, tolerance = 1e-6)))
  })

  test_that("test v2 keeps zero-valued samples in the class size", {
    skip_on_cran()
    # uv run --with 'libauc==2.0.1' --with torch --with 'numpy<2' --python 3.11 python
    #
    #   import torch
    #   from libauc.losses.auc import AUCMLoss
    #   def golden(a, b, al, s_list, y_list, m=1.0, version='v1'):
    #       s = torch.tensor(s_list).view(-1, 1); y = torch.tensor(y_list).view(-1, 1)
    #       p = sum(y_list) / len(y_list)
    #       f = AUCMLoss(margin=m, version=version)
    #       with torch.no_grad(): f.a.fill_(a); f.b.fill_(b); f.alpha.fill_(al)
    #       out = float(f(s, y))
    #       return out / (p * (1 - p)) if version == 'v1' else out
    #   S0 = [0.0, 0.4, 0.35, 0.8]; Y = [0., 0., 1., 1.]
    #   golden(0, 0, 0, S0, Y)         # 0.46125003695487976  LibAUC v2: 0.5412500500679016
    #   golden(0.3, 0.6, 0.5, S0, Y)   # 0.7012500166893005   LibAUC v2: 0.6012499928474426
    pred_with_zero <- torch::torch_tensor(c(0.0, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    expect_equal(
      AUCM(pred_with_zero, label, a = 0, b = 0, alpha = 0, margin = 1, version = "v2")$item(),
      0.4612500,
      tolerance = 1e-6
    )
    expect_equal(
      AUCM(pred_with_zero, label, a = 0.3, b = 0.6, alpha = 0.5, margin = 1, version = "v2")$item(),
      0.7012500,
      tolerance = 1e-6
    )
  })

  test_that("test v2 stays finite when a whole class sits on its center", {
    skip_on_cran()
    #   SAT = [0.0, 0.0, 1.0, 1.0]; Y = [0., 0., 1., 1.]
    #   golden(1.0, 0.0, 0.5, SAT, Y)  # -0.25                LibAUC v2: nan
    pred_saturated <- torch::torch_tensor(c(0, 0, 1, 1))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    saturated <- AUCM(pred_saturated, label,
      a = 1, b = 0, alpha = 0.5, margin = 1, version = "v2"
    )$item()
    expect_false(is.nan(saturated))
    expect_equal(saturated, -0.25, tolerance = 1e-6)
  })

  test_that("test v2 is NaN for a single-class batch while v1 is finite", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    for (label_values in list(c(0, 0, 0, 0), c(1, 1, 1, 1))) {
      label <- torch::torch_tensor(label_values)
      expect_true(is.nan(
        AUCM(pred, label, a = 0, b = 0, alpha = 0, margin = 1, version = "v2")$item()
      ))
      expect_equal(AUCM(pred, label, a = 0, b = 0, alpha = 0, margin = 1)$item(),
        0,
        tolerance = 1e-6
      )
    }
  })

  test_that("test nn_AUCM_loss forwards version to AUCM", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    loss_fn <- nn_AUCM_loss(margin = 1, version = "v2", add_sigmoid=FALSE)
    expect_equal(loss_fn(pred, label)$item(), 0.4662500, tolerance = 1e-6) # gold: see above
    loss_fn_default <- nn_AUCM_loss(margin = 1, add_sigmoid=FALSE)
    expect_equal(loss_fn_default(pred, label)$item(), 0.1165625, tolerance = 1e-6) # gold: see above
  })

  test_that("test v2 gradients of a, b and alpha", {
    skip_on_cran()
    # uv run --with 'libauc==2.0.1' --with torch --with 'numpy<2' --python 3.11 python
    #
    #   import torch
    #   from libauc.losses.auc import AUCMLoss
    #   def grads(a, b, al, s_list, y_list, m=1.0):
    #       s = torch.tensor(s_list).view(-1, 1); y = torch.tensor(y_list).view(-1, 1)
    #       p = sum(y_list) / len(y_list); sc = p * (1 - p)
    #       f = AUCMLoss(margin=m, version='v1')
    #       with torch.no_grad(): f.a.fill_(a); f.b.fill_(b); f.alpha.fill_(al)
    #       L = f(s, y); L.backward()
    #       return [float(f.a.grad)/sc, float(f.b.grad)/sc, float(f.alpha.grad)/sc]
    #   S = [0.1, 0.4, 0.35, 0.8]; Y = [0., 0., 1., 1.]
    #   grads(0, 0, 0, S, Y)        # [-1.149999976158142, -0.5, 1.350000023841858]
    #   grads(0.3, 0.6, 0.5, S, Y)  # [-0.5499999523162842, 0.7000000476837158, 0.3500000238418579]
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    loss_fn <- nn_AUCM_loss(margin = 1, version = "v2", add_sigmoid=FALSE)
    loss <- loss_fn(pred, label)
    expect_true(loss$requires_grad)
    loss$backward()
    expect_equal(as.numeric(loss_fn$a$grad), -1.15, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$b$grad), -0.5, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$alpha$grad), 1.35, tolerance = 1e-6)
    loss_fn2 <- nn_AUCM_loss(margin = 1, version = "v2", add_sigmoid=FALSE)
    torch::with_no_grad({
      loss_fn2$a$fill_(0.3)
      loss_fn2$b$fill_(0.6)
      loss_fn2$alpha$fill_(0.5)
    })
    loss_fn2(pred, label)$backward()
    expect_equal(as.numeric(loss_fn2$a$grad), -0.55, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn2$b$grad), 0.7, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn2$alpha$grad), 0.35, tolerance = 1e-6)
  })

  test_that("test v2 e2e", {
    skip_on_cran()
    set.seed(1)
    torch::torch_manual_seed(1)
    n <- 200
    x1 <- rnorm(n)
    x2 <- rnorm(n)
    y <- factor(ifelse(plogis(1.5 * x1 - x2) > 0.7, "pos", "neg"), levels = c("neg", "pos"))
    task <- mlr3::TaskClassif$new("toy", data.frame(x1, x2, y), target = "y", positive = "pos")
    torch_loss <- mlr3torch::as_torch_loss(nn_AUCM_loss)
    torch_loss$param_set$set_values(margin = 1, version = "v2") # ensure feed the loss probs
    L <- mlr3torch::LearnerTorchMLP$new(task_type = "classif")
    L$loss <- torch_loss
    L$optimizer <- mlr3torch::t_opt("sgd", lr = 0.05)
    L$callbacks <- make_pesg_callback(lr = 0.05)
    L$predict_type <- "prob"
    L$param_set$set_values(epochs = 30, batch_size = 32, neurons = 8, shuffle = FALSE, seed = 1)
    L$train(task)
    expect_equal(L$param_set$values$loss.version, "v2")
    lf <- L$model$loss_fn
    a <- as.numeric(lf$a)
    b <- as.numeric(lf$b)
    alpha <- as.numeric(lf$alpha)
    expect_false(any(is.nan(c(a, b, alpha))))
    expect_gt(a, b)
    expect_gte(alpha, 0)
    expect_gt(L$predict(task)$score(mlr3::msr("classif.auc")), 0.8)
  })

  test_that("test new parameter imratio", {
    skip_on_cran()
    #   uv run --with 'libauc==2.0.1' --with torch --with 'numpy<2' --python 3.11 <script.py>
    #   import torch
    #   from libauc.losses.auc import AUCMLoss
    #   s = torch.tensor([0.1, 0.4, 0.35, 0.8]).view(-1, 1)
    #   y = torch.tensor([0., 0., 1., 1.]).view(-1, 1) # batch p = 0.5
    #   f = AUCMLoss(margin=1.0, imratio=0.01, version='v1')
    #   with torch.no_grad():
    #       f.a.fill_(0.3); f.b.fill_(0.6); f.alpha.fill_(0.5)
    #   float(f(s, y, auto=False)) # --0.2127312421798706
    #   AUCMLoss(margin=1.0, imratio=0.01, version='v1')(s, y, auto=False) # 0.18914376199245453
    #   AUCMLoss(margin=1.0, imratio=0.25, version='v1')(s, y, auto=False) # 0.1535937637090683
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    expect_equal(
      AUCM(pred, label, a = 0, b = 0, alpha = 0, margin = 1, imratio = 0.01)$item(),
      0.1891438,
      tolerance = 1e-6
    )
    expect_equal(
      AUCM(pred, label, a = 0, b = 0, alpha = 0, margin = 1, imratio = 0.25)$item(),
      0.1535938,
      tolerance = 1e-6
    )
    expect_equal(
      AUCM(pred, label,
        a = 0.3, b = 0.6, alpha = 0.5, margin = 1, imratio = 0.01
      )$item(),
      -0.2127312,
      tolerance = 1e-6
    )
  })

  test_that("test new parameter add_sigmoid", {
    skip_on_cran()
    prob <- c(0.1, 0.4, 0.35, 0.8)
    logit <- torch::torch_tensor(qlogis(prob)) # sigmoid(qlogis(p)) == p
    pred <- torch::torch_tensor(prob)
    label <- torch::torch_tensor(c(0, 0, 1, 1))
    expect_equal(
      nn_AUCM_loss(margin = 1)(logit, label)$item(),
      AUCM(pred, label, margin = 1)$item(),
      tolerance = 1e-6
    )
    expect_equal(
      nn_AUCM_loss(margin = 1, version = "v2")(logit, label)$item(),
      AUCM(pred, label, margin = 1, version = "v2")$item(),
      tolerance = 1e-6
    )
    off <- nn_AUCM_loss(margin = 1, add_sigmoid = FALSE)(logit, label)$item()
    expect_false(isTRUE(all.equal(off, AUCM(pred, label, margin = 1)$item(),
      tolerance = 1e-6
    )))
    expect_equal(off, AUCM(logit, label, margin = 1)$item(), tolerance = 1e-6)
    expect_true(nn_AUCM_loss(margin = 1)$add_sigmoid)
    expect_false(nn_AUCM_loss(margin = 1, add_sigmoid = FALSE)$add_sigmoid)
    loss_fn <- nn_AUCM_loss(margin = 1) # add_sigmoid = TRUE by default
    loss <- loss_fn(logit, label) # feed with logits
    expect_true(loss$requires_grad)
    loss$backward()
    grads <- c(
      as.numeric(loss_fn$a$grad), as.numeric(loss_fn$b$grad),
      as.numeric(loss_fn$alpha$grad)
    )
    expect_false(any(is.nan(grads)))
    expect_true(all(is.finite(grads)))
    # Values identical to Step7
    expect_equal(grads, c(-0.2875, -0.125, 0.3375), tolerance = 1e-6)
    scores <- torch::torch_tensor(qlogis(prob), requires_grad = TRUE)
    nn_AUCM_loss(margin = 1)(scores, label)$backward()
    expect_false(any(is.nan(as.numeric(scores$grad))))
    expect_equal(length(as.numeric(scores$grad)), length(prob))
  })
}
