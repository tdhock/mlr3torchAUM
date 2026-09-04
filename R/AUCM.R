AUCM <- function(pred_tensor, label_tensor, a = 0, b = 0, alpha = 0, margin = 1,
                 version = "v1", imratio = NULL) {
  pos_mask <- (label_tensor$flatten() == 1L)$to(torch::torch_float())
  neg_mask <- 1 - pos_mask
  s <- pred_tensor$flatten()
  if (version == "v2") {
    return(
      class_mean((s - a)^2, pos_mask) +
        class_mean((s - b)^2, neg_mask) +
        2 * alpha * (margin + class_mean(s, neg_mask) - class_mean(s, pos_mask)) -
        alpha^2
    )
  }
  p <- imratio
  if (is.null(p)) {
    p <- positive_ratio(label_tensor)
  } else if (p <= 0 || p >= 1) stop("imratio out of range")
  N <- label_tensor$numel()
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

class_mean <- function(x, mask) {
  torch::torch_sum(x * mask) / torch::torch_sum(mask)
}

nn_AUCM_loss <- torch::nn_module(
  c("nn_AUCM_loss", "nn_loss"),
  initialize = function(margin = 1, version = "v1", imratio = NULL, add_sigmoid = TRUE) {
    self$a <- torch::nn_parameter(torch::torch_zeros(1))
    self$b <- torch::nn_parameter(torch::torch_zeros(1))
    self$alpha <- torch::nn_parameter(torch::torch_zeros(1))
    self$margin <- margin
    self$version <- version
    self$imratio <- imratio
    self$add_sigmoid <- add_sigmoid
  },
  forward = function(pred_tensor, label_tensor) {
    if (self$add_sigmoid) pred_tensor <- torch::torch_sigmoid(pred_tensor)
    AUCM(pred_tensor, label_tensor, self$a, self$b, self$alpha, self$margin, self$version, self$imratio)
  }
)

torch_loss_aucm <- function() {
  mlr3torch::TorchLoss$new(
    torch_loss = nn_AUCM_loss,
    task_types = "classif",
    id         = "aucm",
    label      = "LibAUC AUCM min-max margin loss",
    packages   = "mlr3torchAUM",
    man        = "mlr3torchAUM::nn_AUCM_loss"
  )
}
