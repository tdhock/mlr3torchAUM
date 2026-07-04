library(testthat)

test_that("test helper function minority_support_vectors", {
  svm_res <- list(index = c(1L, 2L, 3L, 6L, 8L)) # svm return result: the indices of SV points
  y <- factor(c("maj", "min", "min", "maj", "min", "maj", "maj", "min")) # 1=maj 2=min 3=min 4=maj 5=min 6=maj 7=maj 8=min
  target_class <- "min"
  expect_identical(minority_support_vectors(svm_res$index, y, target_class), c(2L, 3L, 8L))
})

test_that("fit_svm returns SV index", {
  skip_if_not_installed("e1071")
  set.seed(1)
  X <- matrix(rnorm(40), nrow = 20, ncol = 2)
  y <- factor(rep(c("min", "maj"), c(6, 14)))
  m1 <- fit_svm(X, y)
  expect_true(length(m1$index) > 0)
  set.seed(1)
  m2 <- fit_svm(X, y)
  expect_identical(m1$index, m2$index)
})

test_that("test SVMSMOTE fit_resample", {
  skip_if_not_installed("e1071")
  set.seed(1)
  Xmaj <- matrix(rnorm(60, 0, 1), 30, 2)
  Xmin <- matrix(rnorm(40, 0, 1), 20, 2) + matrix(c(2.2, 0), 20, 2, byrow = TRUE)
  X <- rbind(Xmaj, Xmin)
  y <- factor(c(rep("maj", 30), rep("min", 20)))
  s <- SVMSMOTE$new("minority", k_neighbors = 3, m_neighbors = 5, out_step = 0.5)
  set.seed(1)
  r1 <- s$fit_resample(X, y)
  expect_true(is.matrix(r1$X))
  expect_equal(nrow(r1$X), 60)
  expect_equal(as.integer(table(r1$y)["min"]), 30)
  expect_identical(levels(r1$y), levels(y))
  set.seed(1)
  r2 <- s$fit_resample(X, y)
  expect_identical(r1, r2)
})

test_that("edge case: a class whose support vectors are all noise (empty batch)", {
  skip_if_not_installed("e1071")
  set.seed(1)
  Xmaj <- rbind(matrix(rnorm(60, 0, 0.5), 30, 2), matrix(rnorm(60, 0, 0.5), 30, 2) + 5)
  X <- rbind(Xmaj, matrix(c(0, 0, 5, 5), 2, 2, byrow = TRUE))
  y <- factor(c(rep("maj", 60), rep("min", 2)))
  s <- SVMSMOTE$new("minority", k_neighbors = 1, m_neighbors = 5, out_step = 0.5)
  set.seed(1)
  expect_warning(
    res <- s$fit_resample(X, y),
    "all support vectors.*are noise"
  )
  expect_true(is.matrix(res$X))
  expect_equal(nrow(res$X), nrow(X)) # all noise, 0 synthetic samples for "min"
  expect_identical(levels(res$y), levels(y))
})

test_that("edge case: no danger SVs -> whole quota from safe (extrapolation) + message", {
  skip_if_not_installed("e1071")
  set.seed(1)
  Xmaj <- matrix(rnorm(60, 0, 1), 30, 2) + matrix(c(8, 0), 30, 2, byrow = TRUE)
  Xmin <- matrix(rnorm(30, 0, 0.5), 15, 2)
  X <- rbind(Xmaj, Xmin)
  y <- factor(c(rep("maj", 30), rep("min", 15)))
  s <- SVMSMOTE$new("minority", k_neighbors = 5, m_neighbors = 5, out_step = 0.5)
  expect_message(
    res <- s$fit_resample(X, y),
    "no danger support vectors"
  )
  expect_equal(nrow(res$X), nrow(X) + 15L) # 30 - 15 = 15
  expect_equal(as.integer(table(res$y)["min"]), 30)
})

test_that("edge case: no safe SVs -> whole quota from danger (interpolation) + message", {
  skip_if_not_installed("e1071")
  set.seed(1)
  Xmaj <- matrix(rnorm(120, 0, 1), 60, 2)
  ang <- seq(0, 2 * pi, length.out = 13)[1:12]
  Xmin <- cbind(cos(ang), sin(ang)) * 1.9
  X <- rbind(Xmaj, Xmin)
  y <- factor(c(rep("maj", 60), rep("min", 12)))
  s <- SVMSMOTE$new("minority", k_neighbors = 5, m_neighbors = 5, out_step = 0.5)
  expect_message(
    res <- s$fit_resample(X, y),
    "no safe support vectors"
  )
  expect_equal(nrow(res$X), nrow(X) + 48L) # 60 - 12 = 48
  expect_equal(as.integer(table(res$y)["min"]), 60)
})
