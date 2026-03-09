#' Precision-Recall Curve Statistics
#'
#' Computes the precision-recall curve and related statistics from predicted
#' probabilities and binary labels using `torch` tensors.
#'
#' @param pred_tensor A 1D [torch::torch_tensor] of real-valued prediction
#'   scores (higher values indicate stronger evidence for the positive class).
#' @param label_tensor A 1D [torch::torch_tensor] of true class labels,
#'   encoded as `-1` for the negative class and `+1` for the positive class.
#'
#' @details
#' This function computes the recall, precision, False Negative Rate (FNR), 
#' False Discovery Rate (FDR) but also the minimum between the FNR and the FDR. 
#' The min_constant and max_constant give the range of constants which result in
#' the corresponding error values. 
#'
#' @return A named list of [torch::torch_tensor] objects:
#' \itemize{
#'   \item `recall` - recall at each threshold
#'   \item `precision` - precision at each threshold
#'   \item `FNR` - false negative rate
#'   \item `FDR` - false discovery rate
#'   \item `"min(FDR,FNR)"` - minimum of FDR and FNR
#'   \item `min_constant` - left endpoints of the  constant added to predicted score
#'   \item `max_constant` - right endpoints of the  constant added to predicted score
#' }
#'
#' @export

PR_curve <- function(pred_tensor, label_tensor){
  list2env(BaseMetricCalculator()(pred_tensor, label_tensor),
  envir = environment())
  TP = p_total * (1 - FNR)
  FP = n_total * FPR
  FN = p_total * FNR
  precision = torch::torch_where(
  TP + FP == 0,
  torch::torch_tensor(1), 
  TP / (TP + FP))
  recall = TP / (TP + FN)
  FDR = 1 - precision
  list(
    recall=recall,
    precision=precision,
    FNR=FNR,
    FDR=FDR,
    "min(FDR,FNR)"=torch::torch_minimum(FDR, FNR),
    min_constant=min_constant,
    max_constant=max_constant)}
