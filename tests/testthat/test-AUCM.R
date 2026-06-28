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
    loss_fn <- nn_AUCM_loss()
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
    loss_fn <- nn_AUCM_loss() # a=b=alpha=0, margin=1
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
    loss_fn <- nn_AUCM_loss(margin = 1)
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
    loss_fn2 <- nn_AUCM_loss(margin = 0.1)
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
    loss_fn <- nn_AUCM_loss(margin = 1) # a=b=alpha=0
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
    L$loss <- nn_AUCM_loss
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
    lf <- nn_AUCM_loss(margin = 1)
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
    L$loss <- nn_AUCM_loss
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
}
