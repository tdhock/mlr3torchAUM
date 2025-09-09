ROCAUM <- function(pred_tensor, label_tensor){
  N <- NULL
  ## Above to avoid CRAN NOTE.
  is_positive = label_tensor$flatten() == 1
  is_negative = !is_positive
  if(all(as.logical(is_positive)) || all(as.logical(is_negative))){
    return(torch::torch_sum(pred_tensor*0))
  }
  fn_diff = torch::torch_where(is_positive, -1, 0)
  fp_diff = torch::torch_where(is_positive, 0, 1)
  thresh_tensor = -pred_tensor$flatten()
  sorted_indices = torch::torch_argsort(thresh_tensor)
  fp_denom = torch::torch_sum(is_negative) #or 1 for AUM based on count instead of rate
  fn_denom = torch::torch_sum(is_positive) #or 1 for AUM based on count instead of rate
  sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1)/fp_denom
  sorted_fn_cum = -fn_diff[sorted_indices]$flip(1)$cumsum(dim=1)$flip(1)/fn_denom
  sorted_thresh = thresh_tensor[sorted_indices]
  sorted_is_diff = sorted_thresh$diff() != 0
  sorted_fp_end = torch::torch_cat(c(sorted_is_diff, torch::torch_tensor(TRUE)))
  sorted_fn_end = torch::torch_cat(c(torch::torch_tensor(TRUE), sorted_is_diff))
  uniq_thresh = sorted_thresh[sorted_fp_end]
  uniq_fp_after = sorted_fp_cum[sorted_fp_end]
  uniq_fn_before = sorted_fn_cum[sorted_fn_end]
  FPR = torch::torch_cat(c(torch::torch_tensor(0.0), uniq_fp_after))
  FNR = torch::torch_cat(c(uniq_fn_before, torch::torch_tensor(0.0)))
  roc = list(
    FPR=FPR,
    FNR=FNR,
    TPR=1 - FNR,
    "min(FPR,FNR)"=torch::torch_minimum(FPR, FNR),
    min_constant=torch::torch_cat(c(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(c(uniq_thresh, torch::torch_tensor(Inf))))
  min_FPR_FNR = roc[["min(FPR,FNR)"]][2:-2]
  constant_diff = roc$min_constant[2:N]$diff()
  torch::torch_sum(min_FPR_FNR * constant_diff)
}

nn_ROCAUM_loss <- torch::nn_module(
  "nn_ROCAUM_loss",
  inherit = torch::nn_mse_loss,
  initialize = function() {
    super$initialize()
  },
  forward = ROCAUM
)

MeasureClassifROCAUM = R6::R6Class(
  "ROCAUM",
  inherit = mlr3::MeasureClassif,
  public = list(
    initialize = function() { 
      super$initialize(
        id = "classif.rocaum",
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

MeasureClassifInvAUC = R6::R6Class(
  "InvAUC",
  inherit = mlr3::MeasureClassif,
  public = list(
    AUC=mlr3::msr("classif.auc"),
    initialize = function() { 
      super$initialize(
        id = "classif.invauc",
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
      1-self$AUC$score(prediction)
    }
  )
)

