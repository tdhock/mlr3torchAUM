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
    },
    on_begin = function() {
      lrs <- sapply(self$ctx$optimizer$param_groups, function(g) g$lr)
      if(any(lrs != 0)) stop("lr of t_opt should be 0")
    }
  )
}
