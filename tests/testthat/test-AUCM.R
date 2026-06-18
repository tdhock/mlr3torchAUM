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

}
