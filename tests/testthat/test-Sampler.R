library(testthat)

test_that("test BaseSampler and check_sampling_strategy", {
  y <- factor(c(rep("0", 10), rep("1", 50)))
  X <- matrix(rnorm(120), ncol = 2)
  expect_identical(check_sampling_strategy(y, "auto", "over-sampling"), c("0" = 40L))
  expect_error(check_sampling_strategy(y, "unknown", "over-sampling"), "not implemented for over-sampling")
  expect_equal(check_sampling_strategy(y, "auto", "under-sampling"), c("1" = 10L))
  expect_error(check_sampling_strategy(y, "unknown", "under-sampling"), "not implemented for under-sampling")

  sampler <- BaseSampler$new()
  expect_error(sampler$fit_resample(X, y), "not implemented")
  sampler_under <- BaseSampler$new()
  sampler_under$sampling_type <- "under-sampling"
  expect_error(sampler_under$fit_resample(X, y), "not implemented")

  expect_error(BaseSampler$new()$fit_resample(matrix(rnorm(10), ncol = 2), y), "data and label not consistent")
})
