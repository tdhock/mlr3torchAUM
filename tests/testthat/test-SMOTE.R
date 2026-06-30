library(testthat)
data.table::setDTthreads(1L)

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
