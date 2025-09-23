\name{MeasureClassifInvAUC}
\alias{MeasureClassifInvAUC}
\alias{MeasureClassifROCAUM}
\docType{data}
\title{
  mlr3 measures related to Area Under the Minimum (AUM)
}
\description{
  In the context of a mlr3 Learner,
  these measures can be used as the parameters
  \code{measures_train} and
  \code{measures_valid}.
}
\format{R6 Class inherting from \code{\link[mlr3]{MeasureClassif}}.}
\examples{

if(torch::torch_is_installed() && requireNamespace("mlr3torch")){
  data.table::setDTthreads(1L)
  sonar_task <- mlr3::tsk("sonar")
  L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
  L$predict_type <- "prob"
  measure_list <- list(
    mlr3torchAUM::MeasureClassifROCAUM$new(),
    mlr3torchAUM::MeasureClassifInvAUC$new())
  L$loss <- mlr3torchAUM::nn_ROCAUM_loss
  L$callbacks <- mlr3torch::t_clbk("history")
  L$param_set$set_values(
    epochs=5, batch_size=10, seed=1,
    measures_train=measure_list)
  L$train(sonar_task)
  L$model$callbacks$history
}

}

