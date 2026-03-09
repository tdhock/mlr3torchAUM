ROCAUM <- function(pred_tensor, label_tensor){
  N <- NULL
  ## Above to avoid CRAN NOTE.
  list2env(BaseMetricCalculator(function(is_positive){
      if(all(as.logical(is_positive)) || all(as.logical(!is_positive))){
        return(torch::torch_sum(label_tensor*0))}
  })(pred_tensor, label_tensor),envir = environment())
  roc = list(
    FPR=FPR,
    FNR=FNR,
    TPR=1 - FNR,
    "min(FPR,FNR)"=torch::torch_minimum(FPR, FNR),
    min_constant=min_constant,
    max_constant=max_constant)
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2]
  constant_diff = roc$min_constant[2:N]$diff()
  torch::torch_sum(min_FPR_FNR * constant_diff)
}

nn_ROCAUM_loss <- torch::nn_module(
  c("nn_ROCAUM_loss", "nn_loss"),
  initialize = function() {
    for(name in c("evals","zeros","all_one_class")){
      self$buffer(name)
    }
  },
  buffer = function(name, value=torch::torch_tensor(0L)){
    self[[name]] <- torch::nn_buffer(value)
  },
  increment = function(name)self$buffer(name, self[[name]]+1L),
  forward = function(pred_tensor, label_tensor){
    loss_tensor <- ROCAUM(pred_tensor, label_tensor)
    self$increment("evals")
    if(torch::as_array(loss_tensor==0))self$increment("zeros")
    if(torch::as_array((label_tensor[1]==label_tensor)$all()))
      self$increment("all_one_class")
    loss_tensor
  }
)

MeasureClassifROCAUM = R6Class(
  "ROCAUM",
  inherit = MeasureClassif,
  public = list(
    initialize = function() { 
      super$initialize(
        id = "classif.rocaum",
        label = "Area Under Minimum of False Positive and False Negative Rates",
        packages = "torch",
        properties = character(),
        task_properties = "twoclass",
        predict_type = "prob",
        range = c(0, Inf),
        minimize = TRUE
      )
    }
  ),
  private = list(
    .score = function(prediction, ...) {
      pred_tensor <- torch::torch_tensor(prediction$prob[,1])
      label_tensor <- torch::torch_tensor(prediction$truth)
      loss_tensor <- ROCAUM(pred_tensor, label_tensor)
      torch::as_array(loss_tensor)
    }
  )
)

MeasureClassifInvAUC = R6Class(
  "InvAUC",
  inherit = MeasureClassif,
  public = list(
    AUC=msr("classif.auc"),
    initialize = function() { 
      super$initialize(
        id = "classif.invauc",
        label = "1-Area Under ROC Curve",
        packages = "torch",
        properties = character(),
        task_properties = "twoclass",
        predict_type = "prob",
        range = c(0, 1),
        minimize = TRUE
      )
    }
  ),
  private = list(
    .score = function(prediction, ...) {
      1-self$AUC$score(prediction)
    }
  )
)

