library(testthat)

test_that("test RandomOverSampler standard auto mode", {
  y <- factor(c(rep("0", 5), rep("1", 20)))
  X <- matrix(rnorm(50), ncol = 2)

  ros <- RandomOverSampler$new(sampling_strategy = "auto")
  res <- ros$fit_resample(X, y)

  expect_equal(nrow(res$X), 40L)
  expect_equal(as.vector(table(res$y)), c(20L, 20L))
})

test_that("test RandomOverSampler minority mode", {
  y <- factor(c(rep("0", 5), rep("1", 5), rep("2", 20)))
  X <- matrix(rnorm(60), ncol = 2)

  ros <- RandomOverSampler$new(sampling_strategy = "minority")
  res <- ros$fit_resample(X, y)

  expect_equal(as.vector(table(res$y)), c(20L, 5L, 20L))
})

test_that("test RandomOverSampler smoothed bootstrap ROSE", {
  set.seed(42)
  y <- factor(c(rep("0", 10), rep("1", 30)))
  X <- matrix(rnorm(80), ncol = 2)

  ros <- RandomOverSampler$new(sampling_strategy = "auto", shrinkage = 1.0)
  res <- ros$fit_resample(X, y)

  expect_equal(nrow(res$X), 60L)

  # Verify synthetic minority samples (rows 41-60) are perturbed
  syn_samples <- res$X[41:60, ]
  orig_samples <- X[y == "0", ]
  is_duplicate <- any(apply(syn_samples, 1, function(syn_row) {
    any(apply(orig_samples, 1, function(orig_row) all(syn_row == orig_row)))
  }))
  expect_false(is_duplicate)
})

test_that("test RandomOverSampler shrinkage validation error", {
  expect_error(RandomOverSampler$new(shrinkage = -1.0), "non-negative")
})
