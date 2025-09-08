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

if(torch::torch_is_installed() && requireNamespace("mlr3torch")){
  sonar_task <- mlr3::tsk("sonar")
  sonar_task$col_roles$stratum <- "Class"
  L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
  L$param_set$set_values(
    epochs=1, batch_size=10, seed=1,
    batch_sampler=batch_sampler_stratified(1))
  L$train(sonar_task)
}

}
