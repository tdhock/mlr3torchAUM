if (torch::torch_is_installed() && requireNamespace("mlr3torch")) {

  test_that("Step1: AUCMLoss_value returns 0", {
    skip_on_cran()
    pred <- torch::torch_tensor(c(0.5))
    label <- torch::torch_tensor(c(1))
    out <- AUCMLoss(pred, label)
    expect_equal(out$item(), 0, tolerance = 1e-6)
  })

}
