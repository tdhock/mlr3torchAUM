surrogate_loss_names <- c(
  "squared",
  "squared_hinge",
  "hinge",
  "logistic",
  "barrier_hinge"
)

squared_loss <- function(margin, t) {
  (margin - t)^2
}

squared_hinge_loss <- function(margin, t) {
  torch::torch_clamp(margin - t, min = 0)^2
}

hinge_loss <- function(margin, t) {
  torch::torch_clamp(margin - t, min = 0)
}

logistic_loss <- function(scale, t) {
  torch::nnf_softplus(-scale * t)
}

barrier_hinge_loss <- function(margin, scale, t) {
  torch::torch_maximum(
    -scale * (margin + t) + margin,
    torch::torch_maximum(margin - t, scale * (t - margin))
  )
}

get_surrogate_loss <- function(loss_name = "squared_hinge") {
  switch(loss_name,
    squared = squared_loss,
    squared_hinge = squared_hinge_loss,
    hinge = hinge_loss,
    logistic = logistic_loss,
    barrier_hinge = barrier_hinge_loss,
    stop(sprintf(
      "Unknown surrogate loss '%s'. Choose from: %s",
      loss_name,
      paste(surrogate_loss_names, collapse = ", ")
    ), call. = FALSE)
  )
}

nn_pairwise_auc_loss <- torch::nn_module(
  c("PairwiseAUC","nn_loss"),
  initialize = function(
      surr_loss = "logistic",
      margin = 1.0,
      scale = 1.0) {
    self$surr_loss_name <- surr_loss
    self$margin <- margin
    self$scale <- scale
    self$surrogate <- get_surrogate_loss(surr_loss)
    for (name in c("evals", "zeros", "all_one_class")) {
      self$buffer(name)
    }
  },
  buffer = function(name, value = torch::torch_tensor(0L)) {
    self[[name]] <- torch::nn_buffer(value)
  },
  increment = function(name) {
    self$buffer(name, self[[name]] + 1L)
  },
  forward = function(input, target) {
    self$increment("evals")
    if (torch::as_array((target[1] == target)$all())) {
      self$increment("all_one_class")
      self$increment("zeros")
      return(torch::torch_sum(input * 0))
    }
    y_pred <- input$flatten()
    y_true <- target$flatten()
    pos_mask <- y_true == 1L
    neg_mask <- y_true == 0L
    f_ps <- y_pred[pos_mask]
    f_ns <- y_pred[neg_mask]
    t <- f_ps$unsqueeze(2) - f_ns$unsqueeze(1)
    if (self$surr_loss_name == "barrier_hinge") {
      loss <- self$surrogate(self$margin, self$scale, t)
    } else if (self$surr_loss_name == "logistic") {
      loss <- self$surrogate(self$scale, t)
    } else {
      loss <- self$surrogate(self$margin, t)
    }
    loss_val <- loss$mean()
    if (torch::as_array(loss_val == 0)) {
      self$increment("zeros")
    }
    loss_val
  }
)

torch_loss_pairwise_auc <- function() {
  mlr3torch::TorchLoss$new(
    torch_loss = nn_pairwise_auc_loss,
    task_types = "classif",
    id         = "pairwise_auc",
    label      = "Pairwise AUC Loss",
    packages   = "mlr3torchAUM",
    man        = "mlr3torchAUM::nn_pairwise_auc_loss"
  )
}
