nn_weighted_bce_loss <- torch::nn_module(
  c("nn_weighted_bce_loss", "nn_loss"),
  initialize = function(pos_weight = 1) {
    if (is.character(pos_weight)) {
      if (!identical(pos_weight, "auto")) {
        stop('pos_weight must be a positive scalar or "auto"')
      }
    } else if (!is.numeric(pos_weight) || length(pos_weight) != 1 ||
               !is.finite(pos_weight) || pos_weight <= 0) {
      stop('pos_weight must be a positive scalar or "auto"')
    }
    self$pos_weight <- pos_weight
    for (name in c("evals", "all_one_class")) {
      self$buffer(name)
    }
    self$buffer("pw_min", torch::torch_tensor(Inf))
    self$buffer("pw_max", torch::torch_tensor(-Inf))
  },
  buffer = function(name, value = torch::torch_tensor(0L)) {
    self[[name]] <- torch::nn_buffer(value)
  },
  increment = function(name) {
    self$buffer(name, self[[name]] + 1L)
  },
  forward = function(input, target) {
    self$increment("evals")
    dev <- input$device
    target01 <- target$flatten()$to(dtype = torch::torch_float())
    n_pos <- target01$sum()$item()
    n_neg <- target01$numel() - n_pos
    if (n_pos == 0 || n_neg == 0) {
      self$increment("all_one_class")
    }
    w <- if (identical(self$pos_weight, "auto")) {
      if (n_pos == 0 || n_neg == 0) 1 else n_neg / n_pos
    } else {
      self$pos_weight
    }
    w_tensor <- torch::torch_tensor(w, device = dev)
    self$buffer("pw_min", torch::torch_minimum(self$pw_min$to(device = dev), w_tensor))
    self$buffer("pw_max", torch::torch_maximum(self$pw_max$to(device = dev), w_tensor))
    torch::nnf_binary_cross_entropy_with_logits(
      input$flatten(), target01,
      pos_weight = w_tensor
    )
  }
)

torch_loss_weighted_bce <- function() {
  mlr3torch::TorchLoss$new(
    torch_loss = nn_weighted_bce_loss,
    task_types = "classif",
    id         = "weighted_bce",
    label      = "Weighted binary cross-entropy (scalar or auto pos_weight)",
    packages   = "mlr3torchAUM",
    man        = "mlr3torchAUM::nn_weighted_bce_loss"
  )
}
