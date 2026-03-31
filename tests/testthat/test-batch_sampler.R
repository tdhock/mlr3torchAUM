library(testthat)

test_that("random sampler .length matches batch_list length", {
  spam_task <- mlr3::tsk("spam")
  spam_list <- list(task = spam_task)
  
  batch_sampler_class <- mlr3torchAUM::batch_sampler_random(
    batch_size = 32,
    shuffle = FALSE
  )
  batch_sampler_instance <- batch_sampler_class(spam_list)
  
  expect_equal(
    batch_sampler_instance$.length(),
    length(batch_sampler_instance$batch_list)
  )
})