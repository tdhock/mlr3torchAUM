library(testthat)

test_that("test helper function minority_support_vectors", {
  svm_res <- list(index = c(1L, 2L, 3L, 6L, 8L)) # svm return result: the indices of SV points
  y <- factor(c("maj", "min", "min", "maj", "min", "maj", "maj", "min")) # 1=maj 2=min 3=min 4=maj 5=min 6=maj 7=maj 8=min
  target_class <- "min"
  expect_identical(minority_support_vectors(svm_res$index, y, target_class), c(2L, 3L, 8L))
})
