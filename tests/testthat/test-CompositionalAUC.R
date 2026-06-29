# Tests for CompositionalAUCLoss + PDSCA (LibAUC port)

test_that("is_ce_step alternates CE/AUCM on the 2k schedule", {
  # k=1: CE, AUCM, CE, AUCM, ...
  expect_true(is_ce_step(0L, 1L))
  expect_false(is_ce_step(1L, 1L))
  expect_true(is_ce_step(2L, 1L))
  expect_false(is_ce_step(3L, 1L))
  # k=2: CE, CE, AUCM, AUCM, ...
  expect_true(is_ce_step(0L, 2L))
  expect_true(is_ce_step(1L, 2L))
  expect_false(is_ce_step(2L, 2L))
  expect_false(is_ce_step(3L, 2L))
})

if (torch::torch_is_installed() && requireNamespace("mlr3torch")) {
  test_that("nn_CompositionalAUC_loss skeleton", {
    skip_on_cran()
    loss <- nn_CompositionalAUC_loss() # defaults: margin = 1, k = 1
    expect_true(inherits(loss, "nn_CompositionalAUC_loss"))
    expect_true(inherits(loss, "nn_loss"))
    expect_equal(loss$a$item(), 0, tolerance = 1e-6)
    expect_equal(loss$b$item(), 0, tolerance = 1e-6)
    expect_equal(loss$alpha$item(), 0, tolerance = 1e-6)
    expect_equal(length(loss$parameters), 3)
    expect_equal(loss$margin, 1)
    expect_equal(loss$k, 1)
    loss <- nn_CompositionalAUC_loss(k = 2) # defaults: margin = 1
    expect_equal(loss$k, 2)
  })
}
