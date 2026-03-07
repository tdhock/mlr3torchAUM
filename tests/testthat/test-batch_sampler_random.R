library(testthat)
data.table::setDTthreads(1L)

test_that("random sampler: no shuffle, multiple batches, unit test", {
  batch_sampler_class <- mlr3torchAUM::batch_sampler_random(
    batch_size = 4)
  batch_sampler_instance_1 <- batch_sampler_class(
    list(task = list(nrow = 10)))
  batch_list <- batch_sampler_instance_1$batch_list
  expect_equal(length(batch_list), 3)
  expect_equal(unname(lapply(batch_list, length)), list(4, 4, 2))
})
test_that("random sampler: no shuffle, only one batch, unit test", {
  batch_sampler_class <- mlr3torchAUM::batch_sampler_random(
    batch_size = 4)
  batch_sampler_instance_1 <- batch_sampler_class(
    list(task = list(nrow = 4)))
  batch_list <- batch_sampler_instance_1$batch_list
  expect_equal(length(batch_list), 1)
  expect_equal(unname(lapply(batch_list, length)), list(4))
})
