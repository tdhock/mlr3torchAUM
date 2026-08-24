library(testthat)

test_that("BaseUnderSampler abstract class works correctly", {
  y <- factor(c(rep("0", 5), rep("1", 20)))
  X <- matrix(rnorm(50), ncol = 2)
  sampler <- BaseUnderSampler$new()
  expect_equal(sampler$sampling_type, "under-sampling")
  expect_error(sampler$fit_resample(X, y), "not implemented")
})
