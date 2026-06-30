library(testthat)
data.table::setDTthreads(1L)

test_that("test helper function count class", {
  count_result <- count_class(factor(c(rep("0", 100), rep("1", 900))))
  expect_identical(count_result, c("0" = 100L, "1" = 900L))
})
