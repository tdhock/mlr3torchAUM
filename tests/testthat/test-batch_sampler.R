library(testthat)
data.table::setDTthreads(1L)

get_sonar_task <- function() {
  task <- mlr3::tsk("sonar")
  task$filter(208:86) # for imbalance. 111 vs. 12
  return(task)
}

test_that("random sampler: no shuffle, unit test", {
  batch_sampler_class <- mlr3torchAUM::batch_sampler_random(
    batch_size = 4, shuffle = FALSE)
  batch_sampler_instance <- batch_sampler_class(
    list(task = list(nrow = 10)))
  expect_equal(length(batch_sampler_instance$batch_list), 3)
  expect_equal(length(batch_sampler_instance$batch_list[[1]]), 3)
  expect_equal(length(batch_sampler_instance$batch_list[[2]]), 4)
  expect_equal(length(batch_sampler_instance$batch_list[[3]]), 3)
  expect_equal(batch_sampler_instance$batch_list[[1]], c(1,2,3))
  expect_equal(batch_sampler_instance$batch_list[[2]], c(4,5,6,7))
  expect_equal(batch_sampler_instance$batch_list[[3]], c(8,9,10))
})

test_that("random sampler: no shuffle, only one batch, unit test", {
  batch_sampler_class <- mlr3torchAUM::batch_sampler_random(
    batch_size = 4, shuffle = FALSE)
  batch_sampler_instance_1 <- batch_sampler_class(
    list(task = list(nrow = 4)))
  expect_equal(length(batch_sampler_instance_1$batch_list), 2) # strange behavior
  expect_equal(length(batch_sampler_instance_1$batch_list[[1]]), 3)
  expect_equal(batch_sampler_instance_1$batch_list[[1]], c(1,2,3))
  expect_equal(length(batch_sampler_instance_1$batch_list[[2]]), 1)
  expect_equal(batch_sampler_instance_1$batch_list[[2]], c(4))
  batch_sampler_instance_2 <- batch_sampler_class(list(task = list(nrow = 3)))
  expect_equal(length(batch_sampler_instance_2$batch_list), 1)
  expect_equal(length(batch_sampler_instance_2$batch_list[[1]]), 3)
  expect_equal(batch_sampler_instance_2$batch_list[[1]], c(1,2,3))
})

test_that("random sampler: shuffle, unit test", {
  batch_sampler_class <- mlr3torchAUM::batch_sampler_random(
    batch_size = 4, shuffle = TRUE)
  batch_sampler_instance <- batch_sampler_class(
    list(task = list(nrow = 10)))
  batch_list_list <- list()
  for(rep in 1:2){
    if(torch::torch_is_installed()){
    torch::torch_manual_seed(888)}
    else{
    set.seed(888)
    }
    batch_sampler_instance$set_batch_list()
    expect_equal(length(batch_sampler_instance$batch_list), 3)
    expect_equal(length(batch_sampler_instance$batch_list[[1]]), 3)
    expect_equal(length(batch_sampler_instance$batch_list[[2]]), 4)
    expect_equal(length(batch_sampler_instance$batch_list[[3]]), 3)
    batch_list_list[[rep]] <- batch_sampler_instance$batch_list
  }
  expect_identical(batch_list_list[[1]], batch_list_list[[2]])
})

test_that("random sampler: shuffle, length and iteration, unit test", {
  batch_sampler_class <- mlr3torchAUM::batch_sampler_random(
    batch_size = 4, shuffle = TRUE)
  if(torch::torch_is_installed()){
    torch::torch_manual_seed(888)
  }else{
    set.seed(888)
  }
  batch_sampler_instance <- batch_sampler_class(
    list(task = list(nrow = 10)))
  batch_list <- batch_sampler_instance$batch_list
  expect_equal(batch_sampler_instance$.length(), 3)
  batch_iter <- batch_sampler_instance$.iter()
  for(i in 1:3){
    if (i == 3){
      if(torch::torch_is_installed()){
        torch::torch_manual_seed(888)
      }else{
        set.seed(888)
      }
    }
    expect_equal(batch_iter(), batch_list[[i]])
  }
  expect_equal(batch_iter(), coro::exhausted())
  batch_iter <- batch_sampler_instance$.iter()
  for(i in 1:3){
    expect_equal(batch_iter(), batch_list[[i]])
  }
})

test_that("stratified sampler: no shuffle, unit test", {
  sonar_task <- get_sonar_task() # 111 vs. 12
  sonar_task$col_roles$stratum <- "Class"
  Class_vec <- sonar_task$data(sonar_task$row_ids, "Class")$Class
  batch_sampler_class <- mlr3torchAUM::batch_sampler_stratified(
    min_samples_per_stratum = 1, shuffle = FALSE)
  batch_sampler_instance <- batch_sampler_class(
    list(task = sonar_task))
  batch_list <- batch_sampler_instance$batch_list
  expect_equal(batch_list[[1]], c(1,2,3,4,5,6,7,8,9,112))
  batch_size_list <- sapply(batch_list, length)
  expect_equal(sum(batch_size_list), sonar_task$nrow)
  batch_count_mat <- sapply(batch_list, function(i)table(Class_vec[i]))
  expect_equal(sum(batch_count_mat["R",] >= 1), ncol(batch_count_mat))
}
)

test_that("stratified sampler: no shuffle, only one batch, unit test", {
  sonar_task <- get_sonar_task() # 111 vs. 12
  sonar_task$col_roles$stratum <- "Class"
  Class_vec <- sonar_task$data(sonar_task$row_ids, "Class")$Class
  batch_sampler_class <- mlr3torchAUM::batch_sampler_stratified(
    min_samples_per_stratum = 7, shuffle = FALSE)
  batch_sampler_instance <- batch_sampler_class(list(task = sonar_task))
  batch_list <- batch_sampler_instance$batch_list
  expect_equal(length(batch_list), 1)
  expect_equal(length(batch_list[[1]]), sonar_task$nrow)
  expect_equal(batch_list[[1]], 1:sonar_task$nrow)
}
)

test_that("stratified sampler: shuffle, unit test", {
  sonar_task <- get_sonar_task() # 111 vs. 12
  sonar_task$col_roles$stratum <- "Class"
  Class_vec <- sonar_task$data(sonar_task$row_ids, "Class")$Class
  batch_sampler_class <- mlr3torchAUM::batch_sampler_stratified(
    min_samples_per_stratum = 1, shuffle = TRUE)
  batch_sampler_instance <- batch_sampler_class(list(task = sonar_task))
  for(rep in 1:2){
    if(torch::torch_is_installed()){
      torch::torch_manual_seed(888)
    }else{
      set.seed(888)
    }
    batch_sampler_instance$set_batch_list()
    batch_list_list[[rep]] <- batch_sampler_instance$batch_list
  }
  expect_identical(batch_list_list[[1]], batch_list_list[[2]])
})

test_that("stratified sampler: shuffle, length and iteration, unit test", {
  sonar_task <- get_sonar_task() # 111 vs. 12
  sonar_task$col_roles$stratum <- "Class"
  Class_vec <- sonar_task$data(sonar_task$row_ids, "Class")$Class
  batch_sampler_class <- mlr3torchAUM::batch_sampler_stratified(
    min_samples_per_stratum = 1, shuffle = TRUE)
  if(torch::torch_is_installed()){
    torch::torch_manual_seed(888)
  }else{
    set.seed(888)
  }
  batch_sampler_instance <- batch_sampler_class(
    list(task = sonar_task))
  batch_list <- batch_sampler_instance$batch_list
  batch_iter <- batch_sampler_instance$.iter()
  for(i in 1:length(batch_list)){
    if (i == 3){
      if(torch::torch_is_installed()){
        torch::torch_manual_seed(888)
      }else{
        set.seed(888)
      }
    }
    expect_equal(batch_iter(), batch_list[[i]])
  }
  expect_equal(batch_iter(), coro::exhausted())
  batch_iter <- batch_sampler_instance$.iter()
  for(i in 1:length(batch_list)){
    expect_equal(batch_iter(), batch_list[[i]])
  }
})

test_that("stratified sampler: no shuffle and shuffle, error for missing stratum, unit test", {
  sonar_task <- get_sonar_task()
  batch_sampler_class_shuffle <- mlr3torchAUM::batch_sampler_stratified(
    min_samples_per_stratum = 1, shuffle = TRUE)
  batch_sampler_class_no_shuffle <- mlr3torchAUM::batch_sampler_stratified(
    min_samples_per_stratum = 1, shuffle = FALSE)
  expect_error(batch_sampler_class_shuffle(list(task = sonar_task)), 
  "sonar task missing stratum column role")
  expect_error(batch_sampler_class_no_shuffle(list(task = sonar_task)),
  "sonar task missing stratum column role")
})

if(torch::torch_is_installed() && requireNamespace("mlr3torch")){

  test_that("random sampler: shuffle, two learners have same weights, end-to-end test", {
    sonar_task <- get_sonar_task()
    sonar_task$col_roles$stratum <- "Class"
    param_list <- list()
    for(rep_i in 1:2){
      L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
      L$param_set$set_values(
        epochs=1, batch_size=10, seed=1,
        batch_sampler=mlr3torchAUM::batch_sampler_random(10))
      L$train(sonar_task)
      param_list[[rep_i]] <- L$model$network$parameters
    }
    expect_equal(param_list[[1]], param_list[[2]])
  })

  test_that("stratified sampler: shuffle, two learners have same weights, end-to-end test", {
    sonar_task <- get_sonar_task()
    sonar_task$col_roles$stratum <- "Class"
    param_list <- list()
    for(rep_i in 1:2){
      L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
      L$param_set$set_values(
        epochs=1, batch_size=10, seed=1,
        batch_sampler=mlr3torchAUM::batch_sampler_stratified(10))
      L$train(sonar_task)
      param_list[[rep_i]] <- L$model$network$parameters
    }
    expect_equal(param_list[[1]], param_list[[2]])
  })

  test_that("stratified sampler: no shuffle and shuffle, error for missing stratum", {
    sonar_task <- get_sonar_task()
    L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
    for(shuffle in c(TRUE, FALSE)){
      L$param_set$set_values(
        epochs=1, batch_size=10, seed=1,
        batch_sampler=mlr3torchAUM::batch_sampler_stratified(
          1, shuffle=shuffle))
      expect_error({
        L$train(sonar_task)
      }, "sonar task missing stratum column role")
    }
  })
}
