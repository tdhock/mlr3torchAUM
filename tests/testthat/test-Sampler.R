library(testthat)

test_that("test Sampler", {
  y <- factor(c(rep("0", 5), rep("1", 20))) # minor: 5; major: 20
  X <- matrix(rnorm(50), ncol = 2) # 25 samples
  sampler <- BaseSampler$new()
  expect_error(sampler$fit_resample(X, y), "not implemented")
})
