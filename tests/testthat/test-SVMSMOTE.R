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
