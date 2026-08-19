is_ce_step <- function(step, k) {
  return(step %% (2 * k) < k)
}

nn_CompositionalAUC_loss <- torch::nn_module(
  c("nn_CompositionalAUC_loss", "nn_loss"),
  initialize = function(margin = 1, k = 1, add_sigmoid = TRUE) {
    self$a <- torch::nn_parameter(torch::torch_zeros(1))
    self$b <- torch::nn_parameter(torch::torch_zeros(1))
    self$alpha <- torch::nn_parameter(torch::torch_zeros(1))
    self$step <- torch::nn_buffer(torch::torch_zeros(1))
    self$margin <- margin
    self$k <- k
    self$add_sigmoid <- add_sigmoid
  },
  forward = function(pred, target) {
    if (self$add_sigmoid) pred <- torch::torch_sigmoid(pred)
    if (is_ce_step(self$step$item(), self$k)) {
      loss <- torch::nnf_binary_cross_entropy(
        pred$flatten(), target$flatten()
      )
    } else {
      loss <- AUCM(pred, target, self$a, self$b, self$alpha, self$margin)
    }
    self$step$add_(1L)
    return(loss)
  }
)

torch_loss_compositional_auc <- function() {
  mlr3torch::TorchLoss$new(
    torch_loss = nn_CompositionalAUC_loss,
    task_types = "classif",
    id         = "compositional_auc",
    label      = "LibAUC compositional CE/AUCM alternating loss",
    packages   = "mlr3torchAUM",
    man        = "mlr3torchAUM::nn_CompositionalAUC_loss"
  )
}
