is_ce_step <- function(step, k) {
  return(step %% (2 * k) < k)
}

nn_CompositionalAUC_loss <- torch::nn_module(
  c("nn_CompositionalAUC_loss", "nn_loss"),
  initialize = function(margin = 1, k = 1) {
    self$a <- torch::nn_parameter(torch::torch_zeros(1))
    self$b <- torch::nn_parameter(torch::torch_zeros(1))
    self$alpha <- torch::nn_parameter(torch::torch_zeros(1))
    self$step <- torch::nn_buffer(torch::torch_zeros(1))
    self$margin <- margin
    self$k <- k
  },
  forward = function(pred, target) {
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
