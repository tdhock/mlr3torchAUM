test_that("trapz calculates trapezoid area", {
  expect_equal(trapz(c(0, 1), c(1, 1)), 1)
  expect_equal(trapz(c(0, 1), c(0, 1)), 0.5)
  expect_equal(trapz(c(0, 1, 2), c(0, 1, 0)), 1)
})

test_that("get_y_values: Hellinger 得分并排序（对照 Python 原版真值）", {
  y_true <- c(0, 0, 1, 2)
  # one-hot 编码：每行只有真实类别那一列是 1
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
