AUCM <- function(pred_tensor, label_tensor, a = 0, b = 0, alpha = 0, margin = 1) {
  p <- positive_ratio(label_tensor)
  N <- label_tensor$numel()
  pos_mask <- (label_tensor$flatten() == 1L)$to(torch::torch_float())
  neg_mask <- 1 - pos_mask
  s <- pred_tensor$flatten()
  pos_term <- (1 - p) * ((s - a)^2 * pos_mask)$sum() / N
  neg_term <- p * ((s - b)^2 * neg_mask)$sum() / N
  cross <- 2 * alpha * (p * (1 - p) * margin + (p * s * neg_mask - (1 - p) * s * pos_mask)$sum() / N)
  dual <- p * (1 - p) * alpha^2
  return(pos_term + neg_term + cross - dual)
}

positive_ratio <- function(label_tensor) {
  label_tensor <- label_tensor$flatten()
  label_tensor_bool <- label_tensor == 1L
  return(label_tensor_bool$to(torch::torch_float())$mean())
}

nn_AUCM_loss <- torch::nn_module(
  "nn_AUCM_loss",
  initialize = function(margin = 1) {
    self$a <- torch::nn_parameter(torch::torch_zeros(1))
    self$b <- torch::nn_parameter(torch::torch_zeros(1))
    self$alpha <- torch::nn_parameter(torch::torch_zeros(1))
    self$margin <- margin
  },
  forward = function(pred_tensor, label_tensor) {
    AUCM(pred_tensor, label_tensor, self$a, self$b, self$alpha, self$margin)
  }
)
