library(testthat)

test_that("test OverSampler", {
  y <- factor(c(rep("0", 5), rep("1", 20))) # minor: 5; major: 20
  X <- matrix(rnorm(50), ncol = 2) # 25 samples
  over_sampler <- BaseOverSampler$new()
  expect_error(over_sampler$fit_resample(X, y), "not implemented")
})
