library(testthat)
data.table::setDTthreads(1L)

test_that("min 1 spam per batch", {
  spam_task <- mlr3::tsk("spam")
  spam_task$col_roles$stratum <- "type"
  spam_task$filter(1510:4601) # for 10% minority spam class.
  Class_vec <- spam_task$data(spam_task$row_ids, "type")$type
  (count_tab <- table(Class_vec))
  count_tab/sum(count_tab)
  spam_list <- list(task=spam_task)
  batch_sampler_class <- mlr3torchAUM::batch_sampler_stratified(
    min_samples_per_stratum = 1)
  batch_sampler_instance <- batch_sampler_class(spam_list)
  batch_count_mat <- sapply(batch_sampler_instance$batch_list, function(i)table(Class_vec[i]))
  expect_equal(sum(batch_count_mat["spam",] >= 1), ncol(batch_count_mat))
})

if(torch::torch_is_installed() && requireNamespace("mlr3torch")){

  test_that("two learners have same weights", {
    sonar_task <- mlr3::tsk("sonar")
    sonar_task$col_roles$stratum <- "Class"
    param_list <- list()
    for(rep_i in 1:2){
      L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
      L$param_set$set_values(
        epochs=1, batch_size=10, seed=1,
        batch_sampler=mlr3torchAUM::batch_sampler_stratified(1))
      L$train(sonar_task)
      param_list[[rep_i]] <- L$model$network$parameters
    }
    expect_equal(param_list[[1]], param_list[[2]])
  })

  test_that("error for missing stratum", {
    sonar_task <- mlr3::tsk("sonar")
    L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
    L$param_set$set_values(
      epochs=1, batch_size=10, seed=1,
      batch_sampler=mlr3torchAUM::batch_sampler_stratified(1))
    expect_error({
      L$train(sonar_task)
    }, "sonar task missing stratum column role")
  })

}
