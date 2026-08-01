library(testthat)

test_that("test ADASYN validates k_neighbors", {
  X <- matrix(1:10, ncol = 2)
  y <- factor(c(1, 1, 1, 2, 2))
  expect_error(ADASYN$new(k_neighbors = -1)$fit_resample(X, y), "k_neighbors must be a positive integer")
  expect_error(ADASYN$new(k_neighbors = 2.5)$fit_resample(X, y), "k_neighbors must be a positive integer")
  expect_error(ADASYN$new(k_neighbors = "a")$fit_resample(X, y), "k_neighbors must be a positive integer")
})

test_that("test ADASYN assigns more synthetic samples to minority points near decision boundary", {
  set.seed(42)
  safe_min <- matrix(rnorm(20, mean = -10, sd = 0.3), ncol = 2)
  border_min <- matrix(rnorm(20, mean = 2, sd = 0.3), ncol = 2)
  X_maj <- matrix(rnorm(60, mean = 3, sd = 0.5), ncol = 2)
  X <- rbind(safe_min, border_min, X_maj)
  y <- factor(c(rep("min", 20), rep("maj", 30)))

  res <- ADASYN$new(sampling_strategy = "auto", k_neighbors = 3)$fit_resample(X, y)
  syn_X <- res$X[(nrow(X) + 1):nrow(res$X), ]

  expect_true(sum(syn_X[, 1] > -5) > sum(syn_X[, 1] < -5))
})

test_that("test ADASYN fit_resample output shape", {
  set.seed(42)
  X <- rbind(matrix(rnorm(20, mean = 0), ncol = 2), matrix(rnorm(60, mean = 3), ncol = 2))
  y <- factor(c(rep("0", 10), rep("1", 30)))

  res <- ADASYN$new(sampling_strategy = "auto", k_neighbors = 3)$fit_resample(X, y)

  expect_equal(nrow(res$X), length(res$y))
  expect_true(nrow(res$X) > nrow(X))
})

test_that("test ADASYN multi-class oversampling", {
  set.seed(42)
  X <- rbind(
    matrix(rnorm(20, mean = 0, sd = 2), ncol = 2),
    matrix(rnorm(20, mean = 1, sd = 2), ncol = 2),
    matrix(rnorm(80, mean = 0.5, sd = 2), ncol = 2)
  )
  y <- factor(c(rep("a", 10), rep("b", 10), rep("c", 40)))

  res <- ADASYN$new(sampling_strategy = "auto", k_neighbors = 3)$fit_resample(X, y)

  expect_true(sum(res$y == "a") > 10)
  expect_true(sum(res$y == "b") > 10)
  expect_equal(sum(res$y == "c"), 40L)
})

test_that("test ADASYN errors when no neighbors are majority", {
  X <- rbind(
    matrix(c(0, 0, 0, 1, 1, 0, 1, 1), ncol = 2, byrow = TRUE),
    matrix(c(1e6, 0, 1e6, 1, 1e6 + 1, 0, 1e6 + 1, 1,
             1e6 + 2, 0, 1e6 + 2, 1, 1e6 + 3, 0, 1e6 + 3, 1), ncol = 2, byrow = TRUE)
  )
  y <- factor(c(rep("min", 4), rep("maj", 8)))

  expect_error(
    ADASYN$new(sampling_strategy = "auto", k_neighbors = 3)$fit_resample(X, y),
    "Not any neighbors belong to the majority"
  )
})
