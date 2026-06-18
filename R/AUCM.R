AUCM <- function(pred_tensor, label_tensor, a = 0, b = 0) {
  p <- positive_ratio(label_tensor)
  N <- label_tensor$numel()
  pos_mask <- (label_tensor$flatten() == 1L)$to(torch::torch_float())
  neg_mask <- 1 - pos_mask
  pos_term <- (1 - p) * ((pred_tensor$flatten() - a)^2 * pos_mask)$sum() / N
  neg_term <- p * ((pred_tensor$flatten() - b)^2 * neg_mask)$sum() / N
  return(pos_term + neg_term)
}

positive_ratio <- function(label_tensor) {
  label_tensor <- label_tensor$flatten()
  label_tensor_bool <- label_tensor == 1L
  return(label_tensor_bool$to(torch::torch_float())$mean())
}
