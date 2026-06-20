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

pesg_step <- function(loss_module, lr) {
  torch::with_no_grad({
    a <- loss_module$a
    b <- loss_module$b
    alpha <- loss_module$alpha
    a$sub_(lr * a$grad)
    b$sub_(lr * b$grad)
  })
  pesg_alpha_step(loss_module, lr)
  torch::with_no_grad({
    a$grad$zero_()
    b$grad$zero_()
    alpha$grad$zero_()
  })
}

make_pesg_callback <- function(lr_ab = 0.05) {
  mlr3torch::torch_callback(
    "pesg",
    on_after_backward = function() {
      pesg_step(self$ctx$loss_fn, lr_ab)
    }
  )
}
