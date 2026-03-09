#' Precision-Recall Area Under the Curve (PR AUC)
#'
#' Computes the area under the precision-recall curve (PR AUC) using
#' trapezoidal integration of precision with respect to recall.
#'
#' @param pred_tensor A 1D [torch::torch_tensor] of real-valued prediction
#'   scores (higher values indicate stronger evidence for the positive class).
#' @param label_tensor A 1D [torch::torch_tensor] of true class labels,
#'   encoded as `-1` for the negative class and `+1` for the positive class.
#'
#' @details
#' This function calls [PR_curve()] to compute recall and precision values,
#' then applies trapezoidal rule integration to approximate the area under
#' the curve.
#'
#' @return A scalar [torch::torch_tensor] giving the estimated PR AUC value.
#'
#' @seealso [PR_curve()]
#' @export

PR_AUC <- function(pred_tensor, label_tensor){
  pr = PR_curve(pred_tensor, label_tensor)
  N = length(pr$recall)  
  recall_diff = pr$recall[2:N]-pr$recall[1:-2]
  precision_sum = pr$precision[2:N]+pr$precision[1:-2]
  torch::torch_sum(recall_diff*precision_sum/2.0)
}
