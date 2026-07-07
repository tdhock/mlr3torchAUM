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

test_that("median_std: median of per-column population std (ddof = 0)", {
  X <- matrix(c(1, 2, 3, 10, 10, 16, 5, 5, 5), ncol = 3) # 3 * 3, odd med
  expect_equal(median_std(X), sqrt(2 / 3))
  one_col <- matrix(c(1, 2, 3), ncol = 1)
  expect_equal(median_std(one_col), sqrt(2 / 3))
  X <- matrix(c(1, 2, 3, 10, 10, 16), ncol = 2) # 3 * 2 even med
  expect_equal(median_std(X), mean(c(sqrt(2 / 3), sqrt(8))))
  # edge case: single row
  expect_equal(median_std(matrix(c(5, 10, 15), nrow = 1)), 0)
  # edge case: single 1x1 cell
  expect_equal(median_std(matrix(5, 1, 1)), 0)
  # edge case: 0 row or 0 col
  expect_equal(median_std(matrix(numeric(0), 0, 1)), 0)
  expect_equal(median_std(matrix(numeric(0), 1, 0)), 0)
})

test_that("Test smotenc_distances", {
  cont <- matrix(c(0, 6, 0, 6, 0, 6), ncol = 2)
  cat <- matrix(c("a", "a", "b", "x", "y", "x"), ncol = 2)
  D <- smotenc_distances(cont, cat)
  expect_equal(unname(diag(D)), c(0, 0, 0)) # distance to itself is always 0
  expect_true(isSymmetric(D))
  expect_equal(D[1, 2], sqrt((0 - 6)^2 + (6 - 0)^2 + 8 * 1))
  expect_equal(D[1, 3], sqrt(0 + 8 * 1))
  expect_equal(D[2, 3], sqrt((6 - 0)^2 + (0 - 6)^2 + 8 * 2))
  # edge: all categories identical
  expect_equal(
    unname(smotenc_distances(cont, matrix("a", 3, 2))),
    unname(as.matrix(dist(cont)))
  )
  # edge: median_std = 0
  cont0 <- matrix(c(5, 5, 5, 2, 2, 2, 0, 3, 0), ncol = 3)
  expect_equal(
    unname(smotenc_distances(cont0, cat)),
    unname(as.matrix(dist(cont0)))
  )
  # edge: single row
  D1 <- smotenc_distances(matrix(c(1, 2), nrow = 1), matrix("a", nrow = 1))
  expect_equal(dim(D1), c(1L, 1L))
  expect_equal(D1[1, 1], 0)
})

test_that("test make_samples_nc", {
  # 4 points 4 features: 2 cont, 2 categorical
  cont <- matrix(c(0, 0, 10, 0, 0, 10, 10, 10), ncol = 2, byrow = TRUE)
  cat <- matrix(c("a", "x", "a", "y", "b", "x", "b", "y"), ncol = 2, byrow = TRUE)
  nn <- rbind(c(2, 3), c(1, 4), c(1, 4), c(2, 3))
  set.seed(1)
  res <- make_samples_nc(cont, cat, nn, n_to_generate = 6)
  expect_equal(dim(res$continuous), c(6L, 2L))
  expect_equal(dim(res$categorical), c(6L, 2L))
  expect_true(is.numeric(res$continuous))
  expect_true(is.character(res$categorical))
  set.seed(1)
  res2 <- make_samples_nc(cont, cat, nn, 6)
  expect_identical(res, res2)
  expect_true(all(res$categorical[, 1] %in% c("a", "b")))
  expect_true(all(res$categorical[, 2] %in% c("x", "y")))
  cont_only_one_f <- matrix(c(0, 10, 0, 10), ncol = 1)
  cat_only_one_f <- matrix(rep(c("cat", "dog"), each = 4), ncol = 2)
  set.seed(42)
  res_only_one_f <- make_samples_nc(cont_only_one_f, cat_only_one_f, nn, n_to_generate = 5)
  expect_equal(dim(res_only_one_f$categorical), c(5L, 2L)) # 5 points, two categorical features
  # each categorical feature only has one value in neighbors, fixed
  expect_true(all(res_only_one_f$categorical[, 1] == "cat"))
  expect_true(all(res_only_one_f$categorical[, 2] == "dog"))
  # edge case: n_to_generate = 0
  zero <- make_samples_nc(cont, cat, nn, 0)
  expect_equal(dim(zero$continuous), c(0L, 2L))
  expect_equal(dim(zero$categorical), c(0L, 2L))
  # edge case: single categorical feature
  cont_single <- matrix(c(0, 0, 10, 0, 0, 10), ncol = 2, byrow = TRUE)
  nn_single <- rbind(c(2, 3), c(1, 3), c(1, 2))
  cat_single <- matrix(c("a", "a", "b"), ncol = 1)
  set.seed(1)
  r_single <- make_samples_nc(cont_single, cat_single, nn_single, 4)
  expect_equal(dim(r_single$categorical), c(4L, 1L))
  # mode must be over ALL k neighbours, not just the first one:
  # every point's 3 neighbours are (a, b, b) -> mode "b"; grabbing only the first
  # neighbour (the nn_idx[row] vs nn_idx[row, ] slip) would wrongly give "a"
  cont_maj <- matrix(1:5, ncol = 1)
  cat_maj <- matrix(c("a", "b", "b", "b", "a"), ncol = 1)
  nn_maj <- rbind(c(5, 2, 3), c(1, 3, 4), c(1, 2, 4), c(5, 2, 3), c(1, 2, 3))
  set.seed(1)
  r_maj <- make_samples_nc(cont_maj, cat_maj, nn_maj, 8)
  expect_true(all(r_maj$categorical[, 1] == "b"))
})
