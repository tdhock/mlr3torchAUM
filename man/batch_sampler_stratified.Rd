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

  # this is a typical use case.
  sonar_task <- mlr3::tsk("sonar")
  sonar_task$col_roles$stratum <- "Class"
  L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
  L$param_set$set_values(
    epochs=1, batch_size=10, seed=1,
    batch_sampler=mlr3torchAUM::batch_sampler_stratified(1))
  L$train(sonar_task)

  # this is what mlr3torch does internally (users should typically not do this).
  sonar_task <- mlr3::tsk("sonar")
  sonar_task$filter(208:86) # for imbalance.
  (count_tab <- table(sonar_task$data(sonar_task$row_ids, "Class")$Class))
  count_tab/sum(count_tab)
  sonar_task$col_roles$stratum <- "Class"
  sonar_dataset <- mlr3torch::task_dataset(
    sonar_task,
    list(fea=mlr3torch::TorchIngressToken(c("V1","V2"), mlr3torch::batchgetter_num)),
    function(data)torch::torch_tensor(ifelse(data$Class=="M", 1, 0)))
  batch_sampler_class <- mlr3torchAUM::batch_sampler_stratified(
    min_samples_per_stratum = 1)
  torch::torch_manual_seed(5)
  batch_sampler_instance <- batch_sampler_class(sonar_dataset)
  sonar_dl <- torch::dataloader(sonar_dataset, batch_sampler=batch_sampler_instance)
  tab_list <- list()
  coro::loop(for(batch in sonar_dl) {
    tab_list[[length(tab_list)+1]] <- table(torch::as_array(batch$y))
  })
  do.call(rbind, tab_list) # each batch has 1 negative example.
  
}

}
