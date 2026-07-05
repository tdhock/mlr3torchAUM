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
