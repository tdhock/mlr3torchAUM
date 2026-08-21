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

test_that("test helper function count class", {
  count_result <- count_class(factor(c(rep("0", 100), rep("1", 900))))
  expect_identical(count_result, c("0" = 100L, "1" = 900L))
})

test_that("test auto mode of check_sampling strategy (multi-class)", {
  samp_strategy_multi <- check_sampling_strategy(
    factor(c(rep("0", 50), rep("1", 100), rep("2", 1000))),
    strategy = "auto"
  )
  expect_identical(samp_strategy_multi, c("0" = 950L, "1" = 900L))
})

test_that("test minority mode of check_sampling strategy", {
  samp_strategy_multi_mino <- check_sampling_strategy(
    factor(c(rep("0", 50), rep("1", 100), rep("2", 1000))),
    strategy = "minority"
  )
  expect_identical(samp_strategy_multi_mino, c("0" = 950L))
  samp_strategy_multi_mino2 <- check_sampling_strategy(
    factor(c(rep("0", 50), rep("1", 50), rep("2", 1000))),
    strategy = "minority"
  )
  expect_identical(samp_strategy_multi_mino2, c("0" = 950L))
})
