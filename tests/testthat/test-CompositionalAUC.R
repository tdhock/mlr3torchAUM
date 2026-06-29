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
}
