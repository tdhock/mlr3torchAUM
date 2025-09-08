library(testthat)
library(mlr3torchAUM)

run_tests <- torch::torch_is_installed() && requireNamespace("mlr3torch")

if(run_tests){

  test_that("two learners have same weights", {
    sonar_task <- mlr3::tsk("sonar")
    sonar_task$col_roles$stratum <- "Class"
    param_list <- list()
    for(rep_i in 1:2){
      L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
      L$param_set$set_values(
        epochs=1, batch_size=10, seed=1,
        batch_sampler=batch_sampler_stratified(1))
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
      batch_sampler=batch_sampler_stratified(1))
    expect_error({
      L$train(sonar_task)
    }, "sonar task missing stratum column role")
  })

}
