PR_curve <- function(pred_tensor, label_tensor){
  list2env(BaseMetricCalculator()(pred_tensor, label_tensor),
  envir = environment()) # p_total, n_total, FPR, FNR, uniq_thresh
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
    min_constant=torch::torch_cat(list(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(list(uniq_thresh, torch::torch_tensor(Inf))))}
