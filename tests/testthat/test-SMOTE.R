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
})
