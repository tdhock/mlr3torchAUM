library(testthat)

test_that("test split_features", {
  X <- data.frame(
    height = c(160, 170, 180),
    color = c("red", "blue", "red"),
    weight = c(50, 70, 90),
    taste = factor(c("sweet", "salty", "sweet")),
    stringsAsFactors = FALSE
  )
  # case 1: give the legal col indices
  res <- split_features(X, c(2, 4))
  expect_equal(res$continuous, as.matrix(X[, c(1, 3), drop = FALSE])) # height, weight
  expect_true(is.numeric(res$continuous))
  expect_equal(res$categorical, as.matrix(X[, c(2, 4), drop = FALSE])) # color, taste
  expect_true(is.character(res$categorical))
  res_name <- split_features(X, c("color", "taste"))
  expect_equal(colnames(res_name$categorical), c("color", "taste"))
  expect_equal(colnames(res_name$continuous), c("height", "weight"))
  expect_equal(colnames(split_features(X, "auto")$categorical), c("color", "taste"))
  # edge: all-category/all-continuous error
  expect_error(split_features(X[, c(2, 4), drop = FALSE], "auto"), "use SMOTEN instead")
  expect_error(split_features(X[, c(1, 3), drop = FALSE], "auto"), "use SMOTE instead")
  # edge: out of range indices
  expect_error(split_features(X, c(2, 9)), "illegal category name or index")
  expect_error(split_features(X, c("unknown", "taste")), "illegal category name or index")
})
