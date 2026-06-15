test_that("trapz (torch) calculates trapezoid area", {
  expect_equal(trapz(torch::torch_tensor(c(0, 1)),
  torch::torch_tensor(c(1, 1)))$item(), 1, tolerance = 1e-6)
  expect_equal(trapz(torch::torch_tensor(c(0, 1)),
  torch::torch_tensor(c(0, 1)))$item(), 0.5, tolerance = 1e-6)
  expect_equal(trapz(torch::torch_tensor(c(0, 1, 2)),
  torch::torch_tensor(c(0, 1, 0)))$item(), 1, tolerance = 1e-6)
})

test_that("get_y_values (torch): Hellinger score and sort", {
  pred_tensor <- torch::torch_tensor(rbind(
    c(0.7, 0.2, 0.1),
    c(0.3, 0.4, 0.3),
    c(0.2, 0.6, 0.2),
    c(0.1, 0.3, 0.6)
  ))
  one_hot_tensor <- torch::torch_tensor(rbind(
    c(1, 0, 0),
    c(1, 0, 0),
    c(0, 1, 0),
    c(0, 0, 1)
  ))
  label_tensor <- torch::torch_tensor(c(1L, 1L, 2L, 3L),
  dtype = torch::torch_long())
  res <- get_y_values(pred_tensor, one_hot_tensor, label_tensor)
  expect_equal(torch::as_array(res$curve_y),
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

  # 6. labels in random order should also get the same result
  res_shuffled <- map_class_labels(y_true, y_score, labels = c(3, 1, 2))
  expect_equal(res_shuffled$y_true_size, 3)
  expect_equal(res_shuffled$y_true_int_encoded, c(1, 1, 2, 2))
})

test_that("mcp_curve / mcp_score (torch) match python code", {
  pred_tensor <- torch::torch_tensor(rbind(
    c(0.7, 0.2, 0.1),
    c(0.3, 0.4, 0.3),
    c(0.2, 0.6, 0.2),
    c(0.1, 0.3, 0.6)
  ))
  label_tensor <- torch::torch_tensor(c(1L, 1L, 2L, 3L), dtype = torch::torch_long())
  cur <- mcp_curve(pred_tensor, label_tensor)
  expect_equal(torch::as_array(cur$curve_x), c(0, 1/3, 2/3, 1), tolerance = 1e-5)
  expect_equal(torch::as_array(cur$curve_y),
               c(0.327484, 0.525233, 0.525233, 0.595847), tolerance = 1e-5)
  expect_equal(mcp_score(pred_tensor, label_tensor)$item(), 0.504044, tolerance = 1e-5)

  # Error case 1: sample numbers don't match
  expect_error(mcp_curve(pred_tensor,
    torch::torch_tensor(c(1L, 1L, 2L), dtype = torch::torch_long())), "sample")
  # Error case 2/3: certain row's probabilities sum up not to 1
  bad <- torch::torch_tensor(rbind(
    c(0.9, 0.2, 0.1), c(0.3, 0.4, 0.3), c(0.2, 0.6, 0.2), c(0.1, 0.3, 0.6)))
  expect_error(mcp_curve(bad, label_tensor), "probabilities sum")
  bad_low <- torch::torch_tensor(rbind(
    c(0.5, 0.2, 0.1), c(0.3, 0.4, 0.3), c(0.2, 0.6, 0.2), c(0.1, 0.3, 0.6)))
  expect_error(mcp_curve(bad_low, label_tensor), "probabilities sum")
})

test_that("get_class_widths: adjust widths of single sample", {
  # y_true = c(0,0,1,2), 3 classes
  one_hot_tensor <- torch::torch_tensor(rbind(
    c(1, 0, 0),
    c(1, 0, 0),
    c(0, 1, 0),
    c(0, 0, 1)
  ))
  w <- get_class_widths(one_hot_tensor, 3)
  expect_equal(torch::as_array(w), c(0.166667, 0.333333, 0.333333), tolerance = 1e-5)
})

test_that("imcp_curve / imcp_score (torch) match python code", {
  pred_tensor <- torch::torch_tensor(rbind(
    c(0.7, 0.2, 0.1),
    c(0.3, 0.4, 0.3),
    c(0.2, 0.6, 0.2),
    c(0.1, 0.3, 0.6)
  ))
  label_tensor <- torch::torch_tensor(
    c(1L, 1L, 2L, 3L), dtype = torch::torch_long())
  cur <- imcp_curve(pred_tensor, label_tensor)
  expect_equal(torch::as_array(cur$curve_x),
               c(0, 0.083333, 0.333333, 0.666667, 0.916667, 1),
               tolerance = 1e-5)
  expect_equal(torch::as_array(cur$curve_y),
               c(0.327484, 0.327484, 0.525233, 0.525233, 0.595847, 0.595847),
               tolerance = 1e-5)
  expect_equal(imcp_score(pred_tensor, label_tensor)$item(),
               0.498747, tolerance = 1e-5)
})
