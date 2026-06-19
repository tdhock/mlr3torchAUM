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
}
