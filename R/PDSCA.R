pdsca_step <- function(
  loss_module, lr, clamp_value, weight_decay,
  epoch_decay, momentum, pass, state_a = NULL, state_b = NULL
) {
  if (pass == "aucm") {
    res <- pesg_full_step(
      loss_module,
      lr, clamp_value, weight_decay,
      epoch_decay, momentum, state_a, state_b
    )
    state_a <- res$state_a
    state_b <- res$state_b
  }
  return(list(state_a = state_a, state_b = state_b))
}

pdsca_pass <- function(loss_fn) {
  if (is_ce_step(loss_fn$step$item() - 1L, loss_fn$k)) "ce" else "aucm"
}

pdsca_buffer_weight_momentum <- function(p, buffer, weight_momentum) {
  return((1 - weight_momentum) * buffer + weight_momentum * p)
}

pdsca_ce_weight_step <- function(p, grad,
                                 lr0, clamp_value, weight_decay,
                                 epoch_decay, model_ref, weight_momentum, buffer) {
  torch::with_no_grad({
    dp <- pesg_d_p(grad, p, clamp_value, weight_decay, epoch_decay, model_ref)
    p$sub_(lr0 * dp)
    if (is.null(buffer)) {
      buffer_new <- p$clone()
    } else {
      buffer_new <- pdsca_buffer_weight_momentum(p, buffer, weight_momentum)
    }
    p$copy_(buffer_new)
  })
  return(buffer_new)
}

pdsca_weight_step <- function(p, grad, pass,
                              lr, lr0,
                              clamp_value, weight_decay, epoch_decay,
                              weight_momentum, momentum,
                              state) {
  if (is.null(state)) {
    state <- list(
      momentum_buffer = NULL, weight_buffer = NULL,
      model_acc = 0, model_ref = 0, T = 0
    )
  }
  if (pass == "aucm") {
    res <- pesg_primal_step(
      p, grad, lr, clamp_value, weight_decay, epoch_decay,
      state$model_ref, momentum, state$momentum_buffer, state$model_acc
    )
    state$momentum_buffer <- res$buffer
    state$model_acc <- res$model_acc
  } else {
    state$weight_buffer <- pdsca_ce_weight_step(
      p, grad, lr0, clamp_value, weight_decay, epoch_decay,
      state$model_ref, weight_momentum, state$weight_buffer
    )
  }
  return(state)
}

make_pdsca_callback <- function(
  lr, clamp_value, weight_decay,
  epoch_decay, momentum, decay_factor
) {
  state_a <- NULL
  state_b <- NULL
  lr_ab <- lr
  mlr3torch::torch_callback(
    "pdsca",
    on_after_backward = function() {
      res <- pdsca_step(
        self$ctx$loss_fn, lr_ab, clamp_value, weight_decay, epoch_decay,
        momentum, pdsca_pass(self$ctx$loss_fn), state_a, state_b
      )
      state_a <<- res$state_a
      state_b <<- res$state_b
    }
  )
}
