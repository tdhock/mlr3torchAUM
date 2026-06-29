is_ce_step <- function(step, k) {
  return(step %% (2 * k) < k)
}

nn_CompositionalAUC_loss <- torch::nn_module(
  c("nn_CompositionalAUC_loss", "nn_loss"),
  initialize = function(margin = 1, k = 1) {
    self$a <- torch::nn_parameter(torch::torch_zeros(1))
    self$b <- torch::nn_parameter(torch::torch_zeros(1))
    self$alpha <- torch::nn_parameter(torch::torch_zeros(1))
    self$margin <- margin
    self$k <- k
  }
)
