library(testthat)

test_that("test helper function count class", {
  count_result <- count_class(factor(c(rep("0", 100), rep("1", 900))))
  expect_identical(count_result, c("0" = 100L, "1" = 900L))
})

test_that("test auto mode of check_sampling strategy", {
  samp_strategy <- check_sampling_strategy(
    factor(c(rep("0", 100), rep("1", 900))),
    strategy = "auto"
  )
  expect_identical(samp_strategy, c("0" = 800L))
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
  expect_identical(samp_strategy_multi_mino2, c("0" = 950L)) # get just one
})

test_that("test unknown mode of check_sampling strategy", {
  expect_error(check_sampling_strategy(
    factor(c(rep("0", 50), rep("1", 50), rep("2", 1000))),
    strategy = "some unknown strategy"
  ), "not implemented")
})

test_that("test knn_index", {
  data <- matrix(c(0, 0, 1, 0, 5, 5, 6, 5), nrow = 4, byrow = TRUE) # p1 p2 p3 p4
  query <- data[c(1, 3), ] # only check p1=(0,0) and p3=(5,5)；query included in data
  k <- 1
  expect_identical(knn_index(query, data, 1), matrix(c(2L, 4L), ncol = 1))
  X <- matrix(c(0, 0, 1, 0, 5, 0, 6, 0), ncol = 2, byrow = TRUE)
  expect_identical(knn_index(X, X, 1), matrix(c(2L, 1L, 4L, 3L), ncol = 1))
  expect_error(knn_index(X, X, 4), "need at least")
})

test_that("test generate samples", {
  X <- matrix(c(0, 0, 4, 0, 0, 3), ncol = 2, byrow = TRUE)
  nn <- knn_index(X, X, 1) # 3,1,1
  # nn[1,1] for point1 (0,0) and its first neighbor (0,3), with step 0.5 -> (0,0) + 0.5((0,3)-(0,0)) = (0,1.5)
  # nn[2,1] for point2 (4,0) and its first neighbor (0,0) with step 0.25 -> (4,0) + 0.25((0,0)-(4,0)) = (3,0)
  generated_points <- generate_samples(X, nn, rows = c(1, 2), cols = c(1, 1), steps = c(0.5, 0.25))
  expect_equal(generated_points, matrix(c(0, 1.5, 3, 0), ncol = 2, byrow = TRUE))
  generated_points1 <- generate_samples(X, nn, rows = 1, cols = 1, steps = 0.5)
  expect_equal(generated_points1, matrix(c(0, 1.5), nrow = 1))
  X1 <- matrix(c(0, 4, 10), ncol = 1)
  nn1 <- knn_index(X1, X1, 1)
  # (0), neighbor (4), (0)+0.5(4-0) = (2)
  generated_points2 <- generate_samples(X1, nn1, rows = 1, cols = 1, steps = 0.5)
  expect_equal(generated_points2, matrix(c(2), ncol = 1))
})

test_that("test make_samples", {
  X <- matrix(c(0, 0, 4, 0, 0, 3), ncol = 2, byrow = TRUE)
  nn <- knn_index(X, X, 1)
  set.seed(1)
  out <- make_samples(X, nn, 5)
  expect_equal(dim(out), c(5L, 2L))
  set.seed(1)
  a <- make_samples(X, nn, 5)
  set.seed(1)
  b <- make_samples(X, nn, 5)
  expect_identical(a, b)
  set.seed(1)
  a <- make_samples(X, nn, 5)
  set.seed(1)
  b <- make_samples(X, nn, 5, step_size = 1)
  expect_identical(a, b)
  X <- matrix(c(0, 10), ncol = 1)
  nn <- knn_index(X, X, 1)
  set.seed(1)
  out <- make_samples(X, nn, 20, step_size = -0.5)
  expect_true(all(out <= 0 | out > 10))
})

test_that("test BaseSMOTE", {
  y <- factor(c(rep("0", 5), rep("1", 20))) # minor: 5; major: 20
  X <- matrix(rnorm(50), ncol = 2) # 25 samples
  sampler <- BaseSMOTE$new()
  expect_error(sampler$fit_resample(X, y), "not implemented")
})

test_that("test SMOTE", {
  y <- factor(c(rep("0", 5), rep("1", 20))) # minor: 5; major: 20
  X <- matrix(rnorm(50), ncol = 2) # 25 samples
  sampler <- SMOTE$new(sampling_strategy = "auto", k_neighbors = 3)
  set.seed(1)
  res <- sampler$fit_resample(X, y)
  expect_equal(nrow(res$X), 40L) # 25 + 15 = 40
  expect_equal(length(res$y), 40L)
  expect_equal(as.integer(table(res$y)["0"]), 20L)
})
