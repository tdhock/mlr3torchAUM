#' Compute multi-class ROC AUC macro averaged
#'
#' This function computes the multi class ROC AUC using OvR approach and macro
#' averaging. It assumes that all the inputs are torch tensors and labels are
#' in [1,K] with K being the number of classes.
#'
#' @param pred_tensor output of the model assuming it is of dimension NxK
#' (or Nx1 for binary classification)
#' @param label_tensor true labels , tensor of length N
#' @return ROC AUC macro averaged
#'
#' @examplesIf torch::torch_is_installed()
#' \dontrun{
#' # Small example with 3 classes and 10 samples
#' labels = torch::torch_randint(1, 4, size = 10, dtype = torch::torch_long())
#' Draw_ROC_curve_micro(torch::torch_randn(c(10, 3)), labels)
#' }
#' @export
ROC_AUC_macro<-function(pred_tensor,label_tensor){
  n_class=pred_tensor$size(2)
  one_hot_labels = torch::nnf_one_hot(label_tensor, num_classes = n_class)
  is_positive = one_hot_labels
  is_negative =1-one_hot_labels
  fn_diff = -is_positive
  fp_diff = is_negative
  thresh_tensor = -pred_tensor
  fn_denom = is_positive$sum(dim = 1)
  fp_denom = is_negative$sum(dim = 1)
  sorted_indices = torch::torch_argsort(thresh_tensor, dim = 1)
  sorted_fp_cum = torch::torch_gather(fp_diff, dim=1, index=sorted_indices)$cumsum(1)/fp_denom
  sorted_fn_cum = -torch::torch_gather(fn_diff, dim=1, index=sorted_indices)$flip(1)$cumsum(1)$flip(1)/fn_denom
  sorted_thresh = torch::torch_gather(thresh_tensor, dim=1, index=sorted_indices)
  device = pred_tensor$device
  zeros_vec=torch::torch_zeros(1,n_class, device=device)
  FPR = torch::torch_cat(c(zeros_vec, sorted_fp_cum))
  FNR = torch::torch_cat(c(sorted_fn_cum, zeros_vec))
  counts <- torch::torch_bincount(label_tensor, minlength = n_class)
  present <- counts > 0
  sum=torch::torch_trapz(1-FNR, x=FPR, dim=1)
  mean_valid = sum[present]$mean()
}
