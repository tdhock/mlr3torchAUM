# Tests for CompositionalAUCLoss + PDSCA (LibAUC port)

test_that("is_ce_step alternates CE/AUCM on the 2k schedule", {
  # step %% (2*k) < k is TRUE -> CE branch, else AUCM
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
