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
