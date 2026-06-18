AUCM <- function(pred_tensor, label_tensor) {
  device <- pred_tensor$device
  return(torch::torch_zeros(1, device = device))
}

positive_ratio <- function(label_tensor) {
  label_tensor <- label_tensor$flatten()
  label_tensor_bool <- label_tensor == 1L
  return(torch::torch_mean(label_tensor_bool$to(torch::torch_float())))
}
