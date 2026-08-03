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
                                 epoch_decay, model_ref, weight_momentum,
                                 buffer, model_acc) {
  torch::with_no_grad({
    dp <- pesg_d_p(grad, p, clamp_value, weight_decay, epoch_decay, model_ref)
    p$sub_(lr0 * dp)
    if (is.null(buffer)) {
      buffer_new <- p$clone()
    } else {
      buffer_new <- pdsca_buffer_weight_momentum(p, buffer, weight_momentum)
    }
    p$copy_(buffer_new)
    res <- list(buffer = buffer_new, model_acc = model_acc + p)
  })
  return(res)
}

pdsca_weight_step <- function(p, grad, pass,
                              lr0, lr,
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
  } else {
    res <- pdsca_ce_weight_step(
      p, grad, lr0, clamp_value, weight_decay, epoch_decay,
      state$model_ref, weight_momentum, state$weight_buffer, state$model_acc
    )
    state$weight_buffer <- res$buffer
  }
  state$model_acc <- res$model_acc
  return(state)
}

optim_pdsca <- torch::optimizer(
  initialize = function(
    params, lr0, lr, clamp_value, weight_decay,
    epoch_decay, weight_momentum, momentum, decay_factor0, decay_factor
  ) {
    self$loss_ref <- NULL
    super$initialize(params, defaults = list(
      lr0 = lr0, lr = lr, clamp_value = clamp_value,
      weight_decay = weight_decay, epoch_decay = epoch_decay,
      weight_momentum = weight_momentum, momentum = momentum,
      decay_factor0 = decay_factor0, decay_factor = decay_factor
    ))
  },
  step = function() {
    torch::with_no_grad({
      if (is.null(self$loss_ref)) stop("current loss function not set")
      pass <- pdsca_pass(self$loss_ref)
      for (group in self$param_groups) {
        for (param in group$params) {
          if (is.null(param$grad)) next
          state <- self$state$get(param)
          if (is.null(state)) {
            state <- list(
              weight_buffer = NULL,
              momentum_buffer = NULL,
              model_acc = torch::torch_zeros_like(param),
              model_ref = torch::torch_zeros_like(param),
              T = 0
            )
          }
          state <- pdsca_weight_step(
            param, param$grad, pass, group$lr0, group$lr, group$clamp_value,
            group$weight_decay, group$epoch_decay,
            group$weight_momentum, group$momentum, state
          )
          state$T <- state$T + 1 # model_acc already added
          self$state$set(param, state)
        }
      }
    })
  }
)

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
      opt <- self$ctx$optimizer
      if (!("loss_ref" %in% names(opt))) {
        stop(
          "make_pdsca_callback() requires optim_pdsca as the learner's optimizer; ",
          "set L$optimizer <- mlr3torch::as_torch_optimizer(optim_pdsca)."
        )
      }
      opt$loss_ref <- self$ctx$loss_fn
    }
  )
}
