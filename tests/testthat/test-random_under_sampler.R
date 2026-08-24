library(testthat)

test_that("RandomUnderSampler validates replacement parameter", {
  expect_true(RandomUnderSampler$new(replacement = TRUE)$replacement)
  expect_error(RandomUnderSampler$new(replacement = NA), "replacement must be TRUE or FALSE")
})

test_that("RandomUnderSampler auto mode without replacement", {
  y <- factor(c(rep("0", 10), rep("1", 50)))
  X <- matrix(rnorm(120), ncol = 2)

  res <- RandomUnderSampler$new(sampling_strategy = "auto")$fit_resample(X, y)
  expect_equal(nrow(res$X), 20L)
  expect_equal(as.integer(table(res$y)), c(10L, 10L))
})

test_that("RandomUnderSampler all mode with replacement", {
  y <- factor(c(rep("0", 10), rep("1", 50), rep("2", 100)))
  X <- matrix(rnorm(320), ncol = 2)

  set.seed(42)
  res <- RandomUnderSampler$new(sampling_strategy = "all", replacement = TRUE)$fit_resample(X, y)
  expect_equal(nrow(res$X), 30L)
  expect_equal(as.integer(table(res$y)), c(10L, 10L, 10L))
})

test_that("RandomUnderSampler handles data consistency error", {
  y <- factor(c(rep("0", 5), rep("1", 10)))
  X <- matrix(rnorm(20), ncol = 2)
  expect_error(RandomUnderSampler$new()$fit_resample(X, y), "data and label not consistent")
})
