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
  is_positive = label_tensor == 1
  is_negative = label_tensor != 1
  fn_diff = torch::torch_where(is_positive, -1, 0)
  fp_diff = torch::torch_where(is_positive, 0, 1)
  thresh_tensor = -pred_tensor$flatten()
  sorted_indices = torch::torch_argsort(thresh_tensor)
  fp_denom = torch::torch_sum(is_negative) #or 1 for AUM based on count instead of rate
  fn_denom = torch::torch_sum(is_positive) #or 1 for AUM based on count instead of rate
  sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1)/fp_denom
  sorted_fn_cum = -fn_diff[sorted_indices]$flip(dims=1)$cumsum(dim=1)$flip(dims=1)/fn_denom
  sorted_thresh = thresh_tensor[sorted_indices]
  sorted_is_diff = sorted_thresh$diff() != 0
  sorted_fp_end = torch::torch_cat(list(sorted_is_diff, torch::torch_tensor(TRUE)))
  sorted_fn_end = torch::torch_cat(list(torch::torch_tensor(TRUE), sorted_is_diff))
  uniq_thresh = sorted_thresh[sorted_fp_end]
  uniq_fp_after = sorted_fp_cum[sorted_fp_end]
  uniq_fn_before = sorted_fn_cum[sorted_fn_end]
  FPR = torch::torch_cat(list(torch::torch_tensor(0.0), uniq_fp_after))
  FNR = torch::torch_cat(list(uniq_fn_before, torch::torch_tensor(0.0)))
  TP = fn_denom * (1 - FNR)
  FP = fp_denom * FPR
  FN = fn_denom * FNR
  precision = torch::torch_where(
    TP + FP == 0,
    torch::torch_tensor(1), 
    TP / (TP + FP)
  )
  recall = TP / (TP + FN)
  FDR = 1 - precision
  list(
    recall=recall,
    precision=precision,
    FNR=FNR,
    FDR=FDR,
    # 1-precision = False Discovery Rate
    # 1-recall = False Negative Rate
    "min(FDR,FNR)"=torch::torch_minimum(FDR, FNR),
    min_constant=torch::torch_cat(list(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(list(uniq_thresh, torch::torch_tensor(Inf))))
}

