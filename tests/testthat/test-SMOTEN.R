library(testthat)

test_that("test get_feature_wise_mode", {
  neighbors <- matrix(c(
    "red", "red", "red", "blue", # mode: red
    "sweet", "salty", "sweet", "salty" # tie
  ), nrow = 4)
  expect_identical(
    get_feature_wise_mode(neighbors),
    c("red", "salty") # dictionary order, salty < sweet
  )
  # edge case: only one neighbor
  neighbors_only_one <- matrix(c("red", "sweet"), nrow = 1)
  expect_identical(
    get_feature_wise_mode(neighbors_only_one),
    c("red", "sweet")
  )
})

test_that("test make_samples_nominal", {
  set.seed(1)
  X <- matrix(c(
    "red", "sweet", "red", "salty",
    "blue", "salty", "blue", "sweet"
  ), nrow = 4, byrow = TRUE)
  nn_indices <- matrix(c(2, 4, 1, 3, 2, 4, 1, 3), ncol = 2, byrow = TRUE)
  samples1 <- make_samples_nominal(X, nn_indices, 3)
  expect_equal(dim(samples1), c(3, ncol(X))) # n_to_generate * num_features
  set.seed(1)
  samples2 <- make_samples_nominal(X, nn_indices, 3)
  expect_identical(samples1, samples2)
  # edge case: only one sample
  X_one_sample <- matrix(c("red", "sweet"), nrow = 1)
  nn_indices_one_sample <- matrix(1, nrow = 1)
  samples4 <- make_samples_nominal(X_one_sample, nn_indices_one_sample, 3)
  expect_equal(nrow(unique(samples4)), 1L)
  expect_equal(samples4[1, ], X_one_sample[1, ])
  expect_equal(samples4, X_one_sample[rep(1, 3), , drop = FALSE])
  # edge case: no samples
  X_empty <- matrix(character(0), ncol = 2) # num_features = 2
  nn_indices_empty <- matrix(integer(0), ncol = 2) # "k" of knn is 2
  samples5 <- make_samples_nominal(X_empty, nn_indices_empty, 3)
  expect_identical(samples5, matrix(character(0), ncol = 2))
  # edge case: nn_neighbors rows not consistent
  expect_error(make_samples_nominal(X, nn_indices_one_sample, 3), "not consistent")
})
