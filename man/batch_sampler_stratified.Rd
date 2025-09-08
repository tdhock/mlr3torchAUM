\name{batch_sampler_stratified}
\alias{batch_sampler_stratified}
\title{
  Stratified batch sampler
}
\description{
  The value returned by this function can be used as the
  \code{batch_sampler} parameter of a \code{TorchLearner}.
}
\usage{
batch_sampler_stratified(min_samples_per_stratum, shuffle = TRUE)
}
\arguments{
  \item{min_samples_per_stratum}{For each batch, this is the minimum
    number of samples that there will be of each stratum. \code{1L} is
    the smallest possible value. Increasing this value causes batch size
    to increase, proportionally to the distribution of strata.}
  \item{shuffle}{if \code{TRUE} (default), shuffle indices before
    creating batches. Otherwise, each epoch has the same batches, in the
    same order as defined in the data.}
}
\value{
  R6 class created by \code{torch::sampler()}.
}
\author{
Toby Dylan Hocking
}
\examples{

## Imbalanced version of sonar data set.
sonar_task <- mlr3::tsk("sonar")
sonar_task$col_roles$stratum <- "Class"
sonar_task$filter(208:86) # for imbalance.
Class_vec <- sonar_task$data(sonar_task$row_ids, "Class")$Class
(count_tab <- table(Class_vec))
count_tab/sum(count_tab) # ~10% R, ~90% M.

## Not a typical use case, but shows class distribution in each batch.
sonar_list <- list(task=sonar_task)
batch_sampler_class <- mlr3torchAUM::batch_sampler_stratified(
  min_samples_per_stratum = 1)
batch_sampler_instance <- batch_sampler_class(sonar_list)
## In the counts below there is 1 R per batch/column.
sapply(batch_sampler_instance$batch_list, function(i)table(Class_vec[i]))

## Typical use case below is as the batch_sampler param in a TorchLearner.
if(torch::torch_is_installed() && requireNamespace("mlr3torch")){
  L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
  L$param_set$set_values(
    epochs=1, batch_size=10, seed=1,
    batch_sampler=batch_sampler_class)
  L$train(sonar_task)
}

}
