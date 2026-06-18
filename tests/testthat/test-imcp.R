if (torch::torch_is_installed()) {

test_that("trapz (torch) calculates trapezoid area", {
  skip_on_cran()
  expect_equal(trapz(
    torch::torch_tensor(c(0, 1)),
    torch::torch_tensor(c(1, 1))
  )$item(), 1, tolerance = 1e-6)
  expect_equal(trapz(
    torch::torch_tensor(c(0, 1)),
    torch::torch_tensor(c(0, 1))
  )$item(), 0.5, tolerance = 1e-6)
  expect_equal(trapz(
    torch::torch_tensor(c(0, 1, 2)),
    torch::torch_tensor(c(0, 1, 0))
  )$item(), 1, tolerance = 1e-6)
})

test_that("get_y_values (torch): Hellinger score and sort", {
  skip_on_cran()
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
    dtype = torch::torch_long()
  )
  res <- get_y_values(pred_tensor, one_hot_tensor, label_tensor)
  expect_equal(torch::as_array(res$curve_y),
    c(0.327484, 0.525233, 0.525233, 0.595847),
    tolerance = 1e-5
  )
  expect_equal(res$sort_indices, c(2, 3, 4, 1))
})

test_that("mcp_curve / mcp_score (torch) match python code", {
  skip_on_cran()
  pred_tensor <- torch::torch_tensor(rbind(
    c(0.7, 0.2, 0.1),
    c(0.3, 0.4, 0.3),
    c(0.2, 0.6, 0.2),
    c(0.1, 0.3, 0.6)
  ))
  label_tensor <- torch::torch_tensor(c(1L, 1L, 2L, 3L), dtype = torch::torch_long())
  cur <- mcp_curve(pred_tensor, label_tensor)
  expect_equal(torch::as_array(cur$curve_x), c(0, 1 / 3, 2 / 3, 1), tolerance = 1e-5)
  expect_equal(torch::as_array(cur$curve_y),
    c(0.327484, 0.525233, 0.525233, 0.595847),
    tolerance = 1e-5
  )
  expect_equal(mcp_score(pred_tensor, label_tensor)$item(), 0.504044, tolerance = 1e-5)

  # Error case 1: sample numbers don't match
  expect_error(mcp_curve(
    pred_tensor,
    torch::torch_tensor(c(1L, 1L, 2L), dtype = torch::torch_long())
  ), "sample")
  # Error case 2/3: certain row's probabilities sum up not to 1
  bad <- torch::torch_tensor(rbind(
    c(0.9, 0.2, 0.1), c(0.3, 0.4, 0.3), c(0.2, 0.6, 0.2), c(0.1, 0.3, 0.6)
  ))
  expect_error(mcp_curve(bad, label_tensor), "probabilities sum")
  bad_low <- torch::torch_tensor(rbind(
    c(0.5, 0.2, 0.1), c(0.3, 0.4, 0.3), c(0.2, 0.6, 0.2), c(0.1, 0.3, 0.6)
  ))
  expect_error(mcp_curve(bad_low, label_tensor), "probabilities sum")
})

test_that("get_class_widths: adjust widths of single sample", {
  skip_on_cran()
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
  skip_on_cran()
  pred_tensor <- torch::torch_tensor(rbind(
    c(0.7, 0.2, 0.1),
    c(0.3, 0.4, 0.3),
    c(0.2, 0.6, 0.2),
    c(0.1, 0.3, 0.6)
  ))
  label_tensor <- torch::torch_tensor(
    c(1L, 1L, 2L, 3L),
    dtype = torch::torch_long()
  )
  cur <- imcp_curve(pred_tensor, label_tensor)
  expect_equal(torch::as_array(cur$curve_x),
    c(0, 0.083333, 0.333333, 0.666667, 0.916667, 1),
    tolerance = 1e-5
  )
  expect_equal(torch::as_array(cur$curve_y),
    c(0.327484, 0.327484, 0.525233, 0.525233, 0.595847, 0.595847),
    tolerance = 1e-5
  )
  expect_equal(imcp_score(pred_tensor, label_tensor)$item(),
    0.498747,
    tolerance = 1e-5
  )
})

test_that("classif.imcp measure (registered) matches imcp_score", {
  skip_on_cran()
  truth <- factor(c("a", "a", "b", "c"), levels = c("a", "b", "c")) # as.integer -> 1,1,2,3
  prob <- matrix(
    c(
      0.7, 0.3, 0.2, 0.1, # line a
      0.2, 0.4, 0.6, 0.3, # line b
      0.1, 0.3, 0.2, 0.6
    ), # line c
    ncol = 3, dimnames = list(NULL, c("a", "b", "c"))
  )
  p <- mlr3::PredictionClassif$new(row_ids = 1:4, truth = truth, prob = prob)
  expect_equal(unname(p$score(mlr3::msr("classif.imcp"))), 0.498747, tolerance = 1e-5)
})

test_that("classif.mcp measure (registered) matches mcp_score", {
  skip_on_cran()
  truth <- factor(c("a", "a", "b", "c"), levels = c("a", "b", "c"))
  prob <- matrix(
    c(
      0.7, 0.3, 0.2, 0.1,
      0.2, 0.4, 0.6, 0.3,
      0.1, 0.3, 0.2, 0.6
    ),
    ncol = 3, dimnames = list(NULL, c("a", "b", "c"))
  )
  p <- mlr3::PredictionClassif$new(row_ids = 1:4, truth = truth, prob = prob)
  expect_equal(unname(p$score(mlr3::msr("classif.mcp"))), 0.504044, tolerance = 1e-5)
})

test_that("nn_IMCP_loss remains on computational graph", {
  skip_on_cran()
  pred_tensor <- torch::torch_tensor(rbind(
    c(68, 23, 56),
    c(3, 50, 38),
    c(2, 16, 25),
    c(10, 4, 4)
  ), requires_grad=TRUE, 
  dtype = torch::torch_float32())
  label_tensor <- torch::torch_tensor(c(1L,2L,3L,1L),
  dtype = torch::torch_long())
  loss_fn <- nn_IMCP_loss()
  loss <- loss_fn(pred_tensor, label_tensor)
  expect_equal(loss$requires_grad, TRUE)
  loss$backward()
  expect_false(is.null(pred_tensor$grad))
  expect_true(all(is.finite(as.numeric(pred_tensor$grad))))
}) 
}

test_that("nn_IMCP_loss faces confident prediction", {
  skip_on_cran()
  pred_tensor <- torch::torch_tensor(rbind(
    c(6, 2, 256),
    c(3, 5, 338),
    c(2, 160, 2),
    c(10, 4, 4)
  ), requires_grad=TRUE, 
  dtype = torch::torch_float32())
  label_tensor <- torch::torch_tensor(c(3L,3L,2L,1L),
  dtype = torch::torch_long())
  loss_fn <- nn_IMCP_loss()
  loss <- loss_fn(pred_tensor, label_tensor)
  loss$backward()
  expect_true(any(is.finite(as.numeric(pred_tensor$grad))))
}) 
