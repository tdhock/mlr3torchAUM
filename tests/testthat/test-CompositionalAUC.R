# Tests for CompositionalAUCLoss + PDSCA (LibAUC port)

test_that("is_ce_step alternates CE/AUCM on the 2k schedule", {
  # k=1: CE, AUCM, CE, AUCM, ...
  expect_true(is_ce_step(0L, 1L))
  expect_false(is_ce_step(1L, 1L))
  expect_true(is_ce_step(2L, 1L))
  expect_false(is_ce_step(3L, 1L))
  # k=2: CE, CE, AUCM, AUCM, ...
  expect_true(is_ce_step(0L, 2L))
  expect_true(is_ce_step(1L, 2L))
  expect_false(is_ce_step(2L, 2L))
  expect_false(is_ce_step(3L, 2L))
})

if (torch::torch_is_installed() && requireNamespace("mlr3torch")) {
  test_that("nn_CompositionalAUC_loss skeleton", {
    skip_on_cran()
    loss <- nn_CompositionalAUC_loss() # defaults: margin = 1, k = 1
    expect_true(inherits(loss, "nn_CompositionalAUC_loss"))
    expect_true(inherits(loss, "nn_loss"))
    expect_equal(loss$a$item(), 0, tolerance = 1e-6)
    expect_equal(loss$b$item(), 0, tolerance = 1e-6)
    expect_equal(loss$alpha$item(), 0, tolerance = 1e-6)
    expect_equal(length(loss$parameters), 3)
    expect_equal(loss$margin, 1)
    expect_equal(loss$k, 1)
    loss <- nn_CompositionalAUC_loss(k = 2) # defaults: margin = 1
    expect_equal(loss$k, 2)
  })

  test_that("test loss function's CE branch", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8), dtype = torch::torch_float32())
    target <- torch::torch_tensor(c(0, 0, 1, 1), dtype = torch::torch_float32())
    loss_fn <- nn_CompositionalAUC_loss()
    loss <- loss_fn(pred, target)
    # import torch
    # import torch.nn.functional as F
    # yp = torch.tensor([0.1, 0.4, 0.35, 0.8]).reshape(-1, 1)
    # yt = torch.tensor([0., 0., 1., 1.]).reshape(-1, 1)
    # print("BCE step0 =", repr(F.binary_cross_entropy(yp, yt).item()))
    expect_equal(loss$item(), 0.47228795289993286, tolerance = 1e-6)
    expect_equal(loss_fn$step$item(), 1)
  })

  test_that("test loss function", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8), dtype = torch::torch_float32())
    target <- torch::torch_tensor(c(0, 0, 1, 1), dtype = torch::torch_float32())
    loss_fn <- nn_CompositionalAUC_loss()
    loss_fn(pred, target) # first time
    loss2 <- loss_fn(pred, target) # second time, AUCM
    # import torch
    # from libauc.losses.auc import CompositionalAUCLoss
    # loss_fn = CompositionalAUCLoss(margin=1.0, k=1, version='v1', device='cpu')
    # yp = torch.tensor([0.1, 0.4, 0.35, 0.8]).reshape(-1, 1)
    # yt = torch.tensor([0., 0., 1., 1.]).reshape(-1, 1)
    # print("call1 (CE)   =", repr(loss_fn(yp, yt).item()))   # 0.47228795289993286
    # print("call2 (AUCM) =", repr(loss_fn(yp, yt).item()),
    #       "p=", float(loss_fn.p))
    expect_equal(loss2$item(), 0.11656250804662704, tolerance = 1e-6)
    expect_equal(loss_fn$step$item(), 2)
    loss3 <- loss_fn(pred, target) # third time, CE
    expect_equal(loss3$item(), 0.47228795289993286, tolerance = 1e-6)
    expect_equal(loss_fn$step$item(), 3)
    loss4 <- loss_fn(pred, target) # fourth time, AUCM
    expect_equal(loss4$item(), 0.11656250804662704, tolerance = 1e-6)
    expect_equal(loss_fn$step$item(), 4)
  })

  test_that("All one class batch returns no errors no warnings", {
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8),
      dtype = torch::torch_float32()
    )
    target <- torch::torch_tensor(c(0, 0, 0, 0),
      dtype = torch::torch_float32()
    ) # all negative samples
    loss_fn <- nn_CompositionalAUC_loss()
    loss_fn(pred, target) # first time CE
    loss <- loss_fn(pred, target) # second time AUCM
    expect_equal(loss$item(), 0, tolerance = 1e-6)
    pred2 <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8),
      dtype = torch::torch_float32()
    )
    target2 <- torch::torch_tensor(c(1, 1, 1, 1),
      dtype = torch::torch_float32()
    ) # all positive samples
    loss_fn <- nn_CompositionalAUC_loss()
    loss_fn(pred2, target2) # first time CE
    loss2 <- loss_fn(pred2, target2) # second time AUCM
    expect_equal(loss2$item(), 0, tolerance = 1e-6)
  })

  test_that("test pdsca_step", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8),
      dtype = torch::torch_float32(), requires_grad = TRUE
    )
    target <- torch::torch_tensor(c(0, 0, 1, 1), dtype = torch::torch_float32())
    loss_fn <- nn_CompositionalAUC_loss()
    loss_ce <- loss_fn(pred, target) # first time: CE
    loss_ce$backward()
    res_ce <- pdsca_step(
      loss_module = loss_fn, lr = 0.1, clamp_value = 10,
      weight_decay = 0, epoch_decay = 0, momentum = 0.999,
      pass = "ce", state_a = NULL, state_b = NULL
    )
    expect_equal(as.numeric(loss_fn$a), 0)
    expect_equal(as.numeric(loss_fn$b), 0)
    expect_equal(as.numeric(loss_fn$alpha), 0)
    expect_equal(res_ce$state_a, NULL)
    expect_equal(res_ce$state_b, NULL)
    loss_aucm <- loss_fn(pred, target) # second time: AUCM
    loss_aucm$backward()
    res_aucm <- pdsca_step(
      loss_module = loss_fn, lr = 0.1, clamp_value = 10,
      weight_decay = 0, epoch_decay = 0, momentum = 0.999,
      pass = "aucm", state_a = NULL, state_b = NULL
    )
    # verified by LibAUC python library
    expect_equal(as.numeric(loss_fn$a), 0.028750000521540642, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$b), 0.012500000186264515, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$alpha), 0.19675001502037048, tolerance = 1e-6)
  })

  test_that("test pdsca_pass", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8), dtype = torch::torch_float32())
    target <- torch::torch_tensor(c(0, 0, 1, 1), dtype = torch::torch_float32())
    loss_fn <- nn_CompositionalAUC_loss()
    pass_wanted1 <- c("ce", "aucm", "ce", "aucm")
    ce_val <- 0.47228795289993286
    aucm_val <- 0.11656250804662704
    val_wanted1 <- c(ce_val, aucm_val, ce_val, aucm_val)
    for (i in 1:4) {
      loss <- loss_fn(pred, target)
      expect_equal(pdsca_pass(loss_fn), pass_wanted1[i])
      expect_equal(as.numeric(loss), val_wanted1[i], tolerance = 1e-6)
    }
    loss_fn2 <- nn_CompositionalAUC_loss(k = 2)
    pass_wanted2 <- c("ce", "ce", "aucm", "aucm")
    for (i in 1:4) {
      loss <- loss_fn2(pred, target)
      expect_equal(pdsca_pass(loss_fn2), pass_wanted2[i])
    }
  })

  test_that("make_pdsca_callback deals a/b/alpha correctly with pass", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8),
      dtype = torch::torch_float32(), requires_grad = TRUE
    )
    target <- torch::torch_tensor(c(0, 0, 1, 1), dtype = torch::torch_float32())
    cb <- make_pdsca_callback(
      lr = 0.1, clamp_value = 10, weight_decay = 0,
      epoch_decay = 0, momentum = 0.999
    )
    expect_true(inherits(cb, "TorchCallback"))
    cb_inst <- cb$generate()
    loss_fn <- nn_CompositionalAUC_loss()
    cb_inst$ctx <- list(loss_fn = loss_fn)
    loss_ce <- loss_fn(pred, target)
    loss_ce$backward()
    cb_inst$on_after_backward()
    expect_equal(as.numeric(loss_fn$a), 0, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$b), 0, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$alpha), 0, tolerance = 1e-6)
    loss_aucm <- loss_fn(pred, target)
    loss_aucm$backward()
    cb_inst$on_after_backward()
    expect_equal(as.numeric(loss_fn$a), 0.028750000521540642, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$b), 0.012500000186264515, tolerance = 1e-6)
    expect_equal(as.numeric(loss_fn$alpha), 0.19675001502037048, tolerance = 1e-6)
  })

  test_that("test pdsca_buffer_weight_momentum", {
    skip_on_cran()
    buffer <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float32())
    p <- torch::torch_tensor(c(3, 4), dtype = torch::torch_float32())
    weight_momentum <- 0.25
    expect_equal(
      as.numeric(pdsca_buffer_weight_momentum(p, buffer, weight_momentum)),
      c(1.5, 2.5),
      tolerance = 1e-6
    )
    weight_momentum2 <- 0.99
    expect_equal(
      as.numeric(pdsca_buffer_weight_momentum(p, buffer, weight_momentum2)),
      c(2.98, 3.98),
      tolerance = 1e-6
    ) # bigger weight_momentum, less smoother
  })

  test_that("test pdsca_ce_weight_step", {
    skip_on_cran()
    w <- torch::torch_tensor(c(0.3, -0.2), dtype = torch::torch_float32(), requires_grad = TRUE)
    grad <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float32())
    # python LibAUC:
    # w = torch.tensor([0.3,-0.2], requires_grad=True)
    # opt = PDSCA([w], loss_fn, lr=0.1, lr0=0.05, beta1=0.99, beta2=0.999,
    #             weight_decay=0.0, epoch_decay=0.0, clip_value=10.0, device='cpu')
    # each step: loss_fn.alpha.grad = None  # force CE branch
    # (w * torch.tensor([1.,2.])).sum().backward()
    # opt.step()
    expected <- list(
      c(0.25, -0.30000001192092896),
      c(0.2004999965429306, -0.39899998903274536),
      c(0.1509999930858612, -0.49799999594688416),
      c(0.1014999970793724, -0.597000002861023)
    )
    buffer <- NULL
    for (t in seq_along(expected)) {
      buffer <- pdsca_ce_weight_step(
        w, grad,
        lr0 = 0.05, clamp_value = 10, weight_decay = 0,
        epoch_decay = 0, model_ref = 0,
        weight_momentum = 0.99, buffer = buffer
      )
      expect_equal(as.numeric(w), expected[[t]], tolerance = 1e-6)
    }
    w2 <- torch::torch_tensor(c(0.3, -0.2), dtype = torch::torch_float32(), requires_grad = TRUE)
    buffer2 <- NULL
    buffer2 <- pdsca_ce_weight_step(w2, grad, 0.05, 10, 0, 0, 0, 0.99, buffer2)
    expect_equal(as.numeric(w2), c(0.3, -0.2) - 0.05 * c(1, 2), tolerance = 1e-6) # t=0
    before <- as.numeric(w2)
    buffer2 <- pdsca_ce_weight_step(w2, grad, 0.05, 10, 0, 0, 0, 0.99, buffer2)
    expect_equal(as.numeric(w2), before - 0.99 * 0.05 * c(1, 2), tolerance = 1e-6) # t=1
  })

  test_that("PDSCA AUC branch == pesg_primal_step", {
    skip_on_cran()
    #   w = torch.tensor([0.3, -0.2], requires_grad=True)
    #   lf = CompositionalAUCLoss(margin=1.0, k=1, version='v1', device='cpu')
    #   opt = PDSCA([w], lf, lr=0.1, lr0=0.05, beta1=0.99, beta2=0.5,
    #               weight_decay=0.0, epoch_decay=0.0, clip_value=10.0, device='cpu')
    #   for t in range(4):
    #       w.grad = None
    #       lf.alpha.grad = torch.zeros(1)
    #       ((w * torch.tensor([1.,2.])) * (t+1)).sum().backward()
    #       opt.step()
    w <- torch::torch_tensor(c(0.3, -0.2), dtype = torch::torch_float32(), requires_grad = TRUE)
    expected <- list(
      c(0.20000001788139343, -0.4000000059604645),
      c(0.050000011920928955, -0.7000000476837158),
      c(-0.17499999701976776, -1.1500000953674316),
      c(-0.48750001192092896, -1.7750000953674316)
    )
    buffer <- NULL
    model_acc <- torch::torch_zeros(2)
    for (t in seq_along(expected)) {
      grad <- torch::torch_tensor(c(1, 2) * t, dtype = torch::torch_float32())
      res <- pesg_primal_step(
        w, grad,
        lr = 0.1,
        clamp_value = 10, weight_decay = 0, epoch_decay = 0, model_ref = 0,
        momentum = 0.5,
        buffer = buffer, model_acc = model_acc
      )
      buffer <- res$buffer
      model_acc <- res$model_acc
      expect_equal(as.numeric(w), expected[[t]], tolerance = 1e-6)
    }
  })

  test_that("PDSCA pdsca_weight_step", {
    skip_on_cran()
    #   w  = torch.tensor([0.3, -0.2], requires_grad=True)
    #   lf = CompositionalAUCLoss(margin=1.0, k=1, version='v1', device='cpu')
    #   opt = PDSCA([w], lf, lr=0.1, lr0=0.05, beta1=0.99, beta2=0.5,
    #               weight_decay=0.0, epoch_decay=0.0, clip_value=10.0, device='cpu')
    #   for t, ps in enumerate(['ce','auc','ce','auc']):
    #       w.grad = None
    #       lf.alpha.grad = None if ps == 'ce' else torch.zeros(1)
    #       ((w * torch.tensor([1.,2.])) * (t+1)).sum().backward()
    #       opt.step()
    p <- torch::torch_tensor(c(0.3, -0.2), dtype = torch::torch_float32(), requires_grad = TRUE)
    passes <- c("ce", "aucm", "ce", "aucm")
    expected <- list(
      c(0.25, -0.30000001192092896),
      c(0.04999999701976776, -0.7000000476837158),
      c(-0.09650000929832458, -0.9930000305175781),
      c(-0.39650002121925354, -1.593000054359436)
    )
    state <- list(
      weight_buffer = NULL, momentum_buffer = NULL,
      model_acc = torch::torch_zeros(2), 
      model_ref = torch::torch_zeros(2),
      T = 0
    )
    for (t in seq_along(passes)) {
      grad <- torch::torch_tensor(c(1, 2) * t, dtype = torch::torch_float32())
      state <- pdsca_weight_step(
        p, grad, passes[t],
        lr = 0.1, lr0 = 0.05,
        clamp_value = 10, weight_decay = 0, epoch_decay = 0,
        weight_momentum = 0.99, # beta1
        momentum = 0.5, # beta2
        state = state
      )
      expect_equal(as.numeric(p), expected[[t]], tolerance = 1e-6)
    }
    expect_false(is.null(state$weight_buffer))
    expect_false(is.null(state$momentum_buffer))
    expect_equal(as.numeric(state$weight_buffer),
      c(-0.09650000929832458, -0.9930000305175781),
      tolerance = 1e-6
    )
    # 0.5 * 2 + 0.5 * 4 = 1 + 2 = 3
    # 0.5 * 4 + 0.5 * 8 = 2 + 4 = 6
    expect_equal(as.numeric(state$momentum_buffer), c(3, 6), tolerance = 1e-6)
  })

  test_that("PDSCA pdsca_weight_step only CE", {
    skip_on_cran()
    p <- torch::torch_tensor(c(0.3, -0.2), dtype = torch::torch_float32(), requires_grad = TRUE)
    state <- list(
      weight_buffer = NULL, momentum_buffer = NULL,
      model_acc = torch::torch_zeros(2), 
      model_ref = torch::torch_zeros(2),
      T = 0
    )
    grad <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float32())
    state <- pdsca_weight_step(
      p, grad, "ce",
      lr = 0.1, lr0 = 0.05,
      clamp_value = 10, weight_decay = 0, epoch_decay = 0,
      weight_momentum = 0.99, # beta1
      momentum = 0.5, # beta2
      state = state
    )
    expect_false(is.null(state$weight_buffer))
    expect_true(is.null(state$momentum_buffer))
  })

  test_that("PDSCA pdsca_weight_step only AUCM", {
    skip_on_cran()
    p <- torch::torch_tensor(c(0.3, -0.2), dtype = torch::torch_float32(), requires_grad = TRUE)
    state <- list(
      weight_buffer = NULL, momentum_buffer = NULL,
      model_acc = torch::torch_zeros(2), 
      model_ref = torch::torch_zeros(2),
      T = 0
    )
    grad <- torch::torch_tensor(c(1, 2), dtype = torch::torch_float32())
    state <- pdsca_weight_step(
      p, grad, "aucm",
      lr = 0.1, lr0 = 0.05,
      clamp_value = 10, weight_decay = 0, epoch_decay = 0,
      weight_momentum = 0.99, # beta1
      momentum = 0.5, # beta2
      state = state
    )
    expect_true(is.null(state$weight_buffer))
    expect_false(is.null(state$momentum_buffer))
  })
}
