pesg_clamp_grad <- function(grad, clamp_value) {
  return(torch::torch_clamp(grad, -clamp_value, clamp_value))
}

pesg_d_p <- function(
  grad, p, clamp_value, weight_decay,
  epoch_decay = 0, model_ref = 0
) {
  return(pesg_clamp_grad(grad, clamp_value) + weight_decay * p +
    epoch_decay * (p - model_ref))
}

pesg_buffer_momentum <- function(dp, buffer, momentum) {
  return((1 - momentum) * buffer + momentum * dp)
}

pesg_primal_step <- function(p, grad,
                             lr, clamp_value, weight_decay,
                             epoch_decay, model_ref, momentum, buffer, model_acc) {
  torch::with_no_grad({
    dp <- pesg_d_p(grad, p, clamp_value, weight_decay, epoch_decay, model_ref)
    if (momentum != 0) {
      if (is.null(buffer)) {
        buffer_new <- dp$clone()
      } else {
        buffer_new <- pesg_buffer_momentum(dp, buffer, momentum)
      }
    } else {
      buffer_new <- dp # degenerate to naive SGD
    }
    p$sub_(lr * buffer_new)
    res <- list(buffer = buffer_new, model_acc = model_acc + p)
  })
  return(res)
}

pesg_update_regularizer <- function(model_acc, T) {
  return(list(
    model_ref = model_acc / T,
    model_acc = torch::torch_zeros_like(model_acc), T = 0
  ))
}

pesg_update_lr <- function(lr, decay_factor) {
  return(lr / decay_factor)
}

optim_pesg <- torch::optimizer(
  initialize = function(
    params, lr, clamp_value, weight_decay,
    epoch_decay, momentum, decay_factor
  ) {
    super$initialize(params, defaults = list(
      lr = lr, clamp_value = clamp_value, weight_decay = weight_decay,
      epoch_decay = epoch_decay, momentum = momentum, decay_factor = decay_factor
    ))
  },
  step = function() {
    torch::with_no_grad({
      for (group in self$param_groups) {
        for (param in group$params) {
          if (is.null(param$grad)) next
          state <- self$state$get(param)
          if (is.null(state)) {
            state <- list(
              buffer = NULL, # first encounter
              model_acc = torch::torch_zeros_like(param),
              model_ref = torch::torch_zeros_like(param),
              T = 0
            )
          }
          res <- pesg_primal_step(param, param$grad,
            lr = group$lr, clamp_value = group$clamp_value,
            weight_decay = group$weight_decay, epoch_decay = group$epoch_decay, model_ref = state$model_ref,
            momentum = group$momentum, buffer = state$buffer, model_acc = state$model_acc
          )
          state$buffer <- res$buffer
          state$model_acc <- res$model_acc
          state$T <- state$T + 1
          self$state$set(param, state)
        }
      }
    })
  },
  update_regularizer = function() {
    for (group in self$param_groups) {
      for (param in group$params) {
        state <- self$state$get(param)
        res <- pesg_update_regularizer(state$model_acc, state$T)
        state$model_ref <- res$model_ref
        state$model_acc <- res$model_acc
        state$T <- res$T
        self$state$set(param, state)
      }
    }
  },
  update_lr = function() {
    for (i in seq_along(self$param_groups)) {
      self$param_groups[[i]]$lr <- pesg_update_lr(
        self$param_groups[[i]]$lr, self$param_groups[[i]]$decay_factor
      )
    }
  }
)

pesg_alpha_step <- function(loss_module, lr) {
  torch::with_no_grad({
    a <- loss_module$a
    b <- loss_module$b
    alpha <- loss_module$alpha
    margin <- loss_module$margin
    alpha$add_(2 * lr * (margin + b - a - alpha))
    alpha$clamp_(0, 999)
  })
}

pesg_full_step <- function(loss_module,
                           lr, clamp_value, weight_decay,
                           epoch_decay, momentum, state_a, state_b) {
  if (is.null(state_a)) state_a <- list(buffer = NULL, model_ref = 0, model_acc = 0, T = 0)
  if (is.null(state_b)) state_b <- list(buffer = NULL, model_ref = 0, model_acc = 0, T = 0)
  a <- loss_module$a
  b <- loss_module$b
  res_a <- pesg_primal_step(
    a, a$grad, lr, clamp_value, weight_decay,
    epoch_decay, state_a$model_ref, momentum, state_a$buffer, state_a$model_acc
  )
  res_b <- pesg_primal_step(
    b, b$grad, lr, clamp_value, weight_decay,
    epoch_decay, state_b$model_ref, momentum, state_b$buffer, state_b$model_acc
  )
  state_a$buffer <- res_a$buffer
  state_b$buffer <- res_b$buffer
  state_a$model_acc <- res_a$model_acc
  state_b$model_acc <- res_b$model_acc
  state_a$T <- state_a$T + 1
  state_b$T <- state_b$T + 1
  pesg_alpha_step(loss_module, lr)
  torch::with_no_grad({
    a$grad$zero_()
    b$grad$zero_()
    loss_module$alpha$grad$zero_()
  })
  return(list(state_a = state_a, state_b = state_b))
}

pesg_step <- function(loss_module, lr, mode = "sgd", state = NULL, ...) {
  a <- loss_module$a
  b <- loss_module$b
  if (mode == "adam") {
    if (is.null(state)) state <- list(a = NULL, b = NULL)
    state$a <- adam_step(a, lr, state = state$a, ...)
    state$b <- adam_step(b, lr, state = state$b, ...)
  } else {
    torch::with_no_grad({
      a$sub_(lr * a$grad)
      b$sub_(lr * b$grad)
    })
  }
  pesg_alpha_step(loss_module, lr)
  torch::with_no_grad({
    a$grad$zero_()
    b$grad$zero_()
    loss_module$alpha$grad$zero_()
  })
  return(state)
}

make_pesg_callback <- function(lr = 0.05, mode = "sgd", ...) {
  state <- NULL
  mlr3torch::torch_callback(
    "pesg",
    on_after_backward = function() {
      state <<- pesg_step(self$ctx$loss_fn, lr, mode = mode, state = state, ...)
    }
  )
}

make_pesg_callback_full <- function(
  lr, clamp_value, weight_decay,
  epoch_decay, momentum, decay_factor
) {
  state_a <- NULL
  state_b <- NULL
  lr_ab <- lr
  mlr3torch::torch_callback(
    "pesg_full",
    on_after_backward = function() {
      res <- pesg_full_step(
        self$ctx$loss_fn,
        lr_ab, clamp_value, weight_decay,
        epoch_decay, momentum, state_a, state_b
      )
      state_a <<- res$state_a
      state_b <<- res$state_b
    },
    on_epoch_end = function() {
      self$ctx$optimizer$update_regularizer() # network weights
      self$ctx$optimizer$update_lr() # network weights
      reg_a <- pesg_update_regularizer(state_a$model_acc, state_a$T)
      reg_b <- pesg_update_regularizer(state_b$model_acc, state_b$T)
      state_a <<- list(
        model_ref = reg_a$model_ref, model_acc = reg_a$model_acc,
        T = reg_a$T, buffer = state_a$buffer
      )
      state_b <<- list(
        model_ref = reg_b$model_ref, model_acc = reg_b$model_acc,
        T = reg_b$T, buffer = state_b$buffer
      )
      lr_ab <<- pesg_update_lr(lr_ab, decay_factor)
    }
  )
}

adam_step <- function(param, lr = 0.001, betas = c(0.9, 0.999), eps = 1e-8, state = NULL) {
  if (is.null(state)) {
    state <- list(
      m = torch::torch_zeros_like(param),
      v = torch::torch_zeros_like(param),
      t = 0
    )
  }
  state$t <- state$t + 1
  grad <- param$grad
  state$m <- betas[1] * state$m + (1 - betas[1]) * grad
  state$v <- betas[2] * state$v + (1 - betas[2]) * grad^2
  m_hat <- state$m / (1 - betas[1]^state$t)
  v_hat <- state$v / (1 - betas[2]^state$t)
  torch::with_no_grad(param$sub_(lr * m_hat / (torch::torch_sqrt(v_hat) + eps)))
  return(state)
}
