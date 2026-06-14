test_that("trapz calculates trapezoid area", {
  expect_equal(trapz(c(0, 1), c(1, 1)), 1)
  expect_equal(trapz(c(0, 1), c(0, 1)), 0.5)
  expect_equal(trapz(c(0, 1, 2), c(0, 1, 0)), 1)
})

test_that("get_y_values: Hellinger score and sort (according to python code)", {
  y_true <- c(0, 0, 1, 2)
  y_true_score <- rbind(
    c(1, 0, 0),
    c(1, 0, 0),
    c(0, 1, 0),
    c(0, 0, 1)
  )
  y_score <- rbind(
    c(0.7, 0.2, 0.1),
    c(0.3, 0.4, 0.3),
    c(0.2, 0.6, 0.2),
    c(0.1, 0.3, 0.6)
  )
  res <- get_y_values(y_true, y_true_score, y_score)
  expect_equal(res$curve_y,
               c(0.327484, 0.525233, 0.525233, 0.595847),
               tolerance = 1e-5)
  expect_equal(res$sort_indices, c(2, 3, 4, 1))
})

test_that("map_class_labels: in normal case, map labels to 1..k", {
  y_score3 <- matrix(1/3, nrow = 4, ncol = 3)
  res1 <- map_class_labels(c(0, 0, 1, 2), y_score3)
  expect_equal(res1$y_true_size, 3)
  expect_equal(res1$y_true_int_encoded, c(1, 1, 2, 3))

  res2 <- map_class_labels(c("b", "b", "a", "c"), y_score3)
  expect_equal(res2$y_true_size, 3)
  expect_equal(res2$y_true_int_encoded, c(2, 2, 1, 3))
})

test_that("map_class_labels: check when cols of y_score is not equal to the number of classes", {
  # y_true only has 2 classes but y_score has 3, must provide labels
  y_true <- c(1, 1, 2, 2)
  y_score <- matrix(1/3, nrow = 4, ncol = 3)

  # 1. no labels provided
  expect_error(map_class_labels(y_true, y_score), "not given")
  # 2. labels length is not right
  expect_error(map_class_labels(y_true, y_score, labels = c(1, 2)), "not equal")
  expect_error(map_class_labels(y_true, y_score, labels = c(1, 2, 3, 4)), "not equal")
  # 3. types are not corret: numeric y_true with characters labels
  expect_error(map_class_labels(y_true, y_score, labels = c(1, 2, "d")), "type does not match")
  # 4. y_true classes are not a subset of labels (missing 2)
  expect_error(map_class_labels(y_true, y_score, labels = c(1, 3, 4)), "subset")
  # 5. provide all c(1,2,3), should succeed
  res <- map_class_labels(y_true, y_score, labels = c(1, 2, 3))
  expect_equal(res$y_true_size, 3)
  expect_equal(res$y_true_int_encoded, c(1, 1, 2, 2))

  # 6. labels in random order should also get the same result (because internal sort)
  res_shuffled <- map_class_labels(y_true, y_score, labels = c(3, 1, 2))
  expect_equal(res_shuffled$y_true_size, 3)
  expect_equal(res_shuffled$y_true_int_encoded, c(1, 1, 2, 2))
})
