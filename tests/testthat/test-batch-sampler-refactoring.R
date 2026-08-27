library(testthat)
data.table::setDTthreads(1L)

skip_if_not(torch::torch_is_installed())

test_that("batch samplers correctly inherit from BaseBatchSampler", {
  task <- mlr3::tsk("sonar")
  task$col_roles$stratum <- "Class"
  rand_sampler <- mlr3torchAUM::batch_sampler_random(batch_size = 10, shuffle = FALSE)(list(task = task))
  expect_true(inherits(rand_sampler, "RandomSampler"))
  expect_true(inherits(rand_sampler, "BaseBatchSampler"))
  strat_sampler <- mlr3torchAUM::batch_sampler_stratified(min_samples_per_stratum = 1)(list(task = task))
  expect_true(inherits(strat_sampler, "StratifiedSampler"))
  expect_true(inherits(strat_sampler, "BaseBatchSampler"))
})
