#' Proposed AUM (Area Under Minimum)
#'
#' Computes the proposed Area Under the Curve of the minimum between
#' False Discovery Rate (FDR) and False Negative Rate (FNR), based on
#' the precision-recall curve.
#'
#' @param pred_tensor A 1D [torch::torch_tensor] of real-valued prediction
#'   scores (higher values indicate stronger evidence for the positive class).
#' @param label_tensor A 1D [torch::torch_tensor] of true class labels,
#'   encoded as `-1` for the negative class and `+1` for the positive class.
#'
#' @details
#' This measure integrates the minimum of FDR and FNR across
#' threshold intervals induced by prediction scores.
#' 
#' @return A scalar [torch::torch_tensor] containing the proposed AUM value.
#'
#' @seealso [PR_curve()]
#' @export

PR_AUM <- function(pred_tensor, label_tensor){
  pr = PR_curve(pred_tensor, label_tensor)
  N = length(pr$recall)  
  min_FDR_FNR = pr[["min(FDR,FNR)"]][2:-2]
  constant_diff = pr$min_constant[2:N]$diff()
  torch::torch_sum(min_FDR_FNR * constant_diff)
}
