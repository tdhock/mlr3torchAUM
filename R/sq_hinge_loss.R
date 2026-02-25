##' All-Pairs Squared Hinge Loss (log-linear time via cumsum)
##'
##' Implements the squared hinge all-pairs loss from Rust & Hocking (2023)
##' https://arxiv.org/abs/2302.11062
##'
##' For a pair (j positive, k negative) the loss is:
##'   l(y_hat_j - y_hat_k) = max(0, m - (y_hat_j - y_hat_k))^2
##' where m >= 0 is the margin hyperparameter (default 1).
##'
##' @param pred  Numeric vector of predicted scores (or torch 1-D tensor).
##' @param label Integer vector with labels +1 (positive) and -1 (negative).
##' @param margin Non-negative margin hyperparameter (default 1).
##' @return A scalar torch tensor (with grad_fn if \code{pred} requires grad).
##' @examples
##' library(torch)
##' pred  <- torch_tensor(c(-1.0, 0.5, 1.2, -0.3), requires_grad = TRUE)
##' label <- c(-1L, 1L, 1L, -1L)
##' loss  <- all_pairs_squared_hinge_loss(pred, label)
##' loss$backward()
##' pred$grad
##' @export
all_pairs_squared_hinge_loss <- function(pred, label, margin = 1) {
  if (!inherits(pred, "torch_tensor")) {
    pred <- torch::torch_tensor(pred, dtype = torch::torch_float())
  }
  label <- as.integer(label)
  stopifnot(length(label) == pred$shape[1], all(label %in% c(-1L, 1L)))
  
  n      <- pred$shape[1]
  is_neg <- (label == -1L)
  
  ## Augment negative predictions by margin, sort ascending (no grad needed)
  torch::with_no_grad({
    aug_offset <- torch::torch_zeros(n, dtype = pred$dtype, device = pred$device)
    if (any(is_neg)) aug_offset[is_neg] <- margin
    sorted_idx <- torch::torch_argsort(pred$detach() + aug_offset)
  })
  
  sorted_idx_r <- as.integer(sorted_idx)
  label_sorted <- label[sorted_idx_r]
  
  ## Walk sorted order: accumulate quadratic coefficients for positives,
  ## evaluate quadratic at each negative (Algorithm 2 from the paper).
  a_val <- 0.0
  b_ten <- torch::torch_tensor(0.0, dtype = pred$dtype, device = pred$device)
  c_ten <- torch::torch_tensor(0.0, dtype = pred$dtype, device = pred$device)
  loss  <- torch::torch_tensor(0.0, dtype = pred$dtype, device = pred$device)
  
  for (i in seq_len(n)) {
    pred_i <- pred[sorted_idx_r[i]]
    if (label_sorted[i] == 1L) {
      z     <- margin - pred_i
      a_val <- a_val + 1.0
      b_ten <- b_ten + 2.0 * z
      c_ten <- c_ten + z * z
    } else if (a_val > 0.0) {
      loss <- loss + a_val * pred_i * pred_i + b_ten * pred_i + c_ten
    }
  }
  
  loss$squeeze()
}


##' Vectorised (cumsum-based) All-Pairs Squared Hinge Loss
##'
##' A fully vectorised version of \code{all_pairs_squared_hinge_loss} that
##' avoids an R-level for-loop by using \code{torch_cumsum}.
##'
##' @inheritParams all_pairs_squared_hinge_loss
##' @return A scalar torch tensor.
##' @export
all_pairs_squared_hinge_loss_vec <- function(pred, label, margin = 1) {
  if (!inherits(pred, "torch_tensor")) {
    pred <- torch::torch_tensor(as.numeric(pred), dtype = torch::torch_float())
  }
  label <- as.integer(label)
  stopifnot(length(label) == pred$shape[1], all(label %in% c(-1L, 1L)))
  
  n      <- pred$shape[1]
  is_neg <- (label == -1L)
  
  ## Augment & sort
  aug <- torch::torch_zeros(n, dtype = torch::torch_float(), device = pred$device)
  if (any(is_neg)) aug[is_neg] <- margin
  torch::with_no_grad({
    sorted_idx <- torch::torch_argsort(pred$detach() + aug)
  })
  
  pred_s  <- pred[sorted_idx]
  label_s <- label[as.integer(sorted_idx)]
  
  is_pos_s <- torch::torch_tensor(as.integer(label_s == 1L),
                                  dtype = pred$dtype, device = pred$device)
  is_neg_s <- 1.0 - is_pos_s
  
  ## Cumulative coefficients (only accumulate for positive examples)
  z     <- (margin - pred_s) * is_pos_s
  cum_a <- torch::torch_cumsum(is_pos_s, dim = 1)
  cum_b <- torch::torch_cumsum(2.0 * z,  dim = 1)
  cum_c <- torch::torch_cumsum(z * z,    dim = 1)
  
  ## Shift by one: each negative uses coefficients from positives before it
  zero   <- torch::torch_zeros(1, dtype = pred$dtype, device = pred$device)
  a_prev <- torch::torch_cat(list(zero, cum_a[1:(n - 1)]))
  b_prev <- torch::torch_cat(list(zero, cum_b[1:(n - 1)]))
  c_prev <- torch::torch_cat(list(zero, cum_c[1:(n - 1)]))
  
  contrib <- is_neg_s * (a_prev * pred_s * pred_s + b_prev * pred_s + c_prev)
  torch::torch_sum(contrib)$squeeze()
}


##' Naive O(n^2) All-Pairs Squared Hinge Loss (reference / testing only)
##'
##' @inheritParams all_pairs_squared_hinge_loss
##' @return A scalar torch tensor.
##' @export
all_pairs_squared_hinge_loss_naive <- function(pred, label, margin = 1) {
  if (!inherits(pred, "torch_tensor")) {
    pred <- torch::torch_tensor(as.numeric(pred), dtype = torch::torch_float())
  }
  label   <- as.integer(label)
  pos_idx <- which(label ==  1L)
  neg_idx <- which(label == -1L)
  
  if (length(pos_idx) == 0L || length(neg_idx) == 0L) {
    return(torch::torch_tensor(0.0, dtype = pred$dtype, device = pred$device))
  }
  
  total <- torch::torch_tensor(0.0, dtype = pred$dtype, device = pred$device)
  for (j in pos_idx) for (k in neg_idx) {
    hinge <- torch::torch_clamp(margin - (pred[j] - pred[k]), min = 0.0)
    total <- total + hinge * hinge
  }
  total
}


##' mlr3torch nn_module wrapper: All-Pairs Squared Hinge Loss
##'
##' Use as \code{loss = t_loss("sq_hinge_pairs")} in any \code{LearnerTorch}.
##' @export
nn_sq_hinge_pairs_loss <- torch::nn_module(
  "nn_sq_hinge_pairs_loss",
  initialize = function(margin = 1.0) self$margin <- margin,
  forward = function(input, target) {
    all_pairs_squared_hinge_loss_vec(
      pred   = input$squeeze(),
      label  = as.integer(target$squeeze()$cpu()),
      margin = self$margin
    )
  }
)

## Register with mlr3torch if available
if (requireNamespace("mlr3torch", quietly = TRUE)) {
  mlr3torch::mlr3torch_losses$add(
    "sq_hinge_pairs",
    mlr3torch::TorchLoss$new(
      torch_loss = nn_sq_hinge_pairs_loss,
      task_types = "classif",
      param_set  = paradox::ps(margin = paradox::p_dbl(lower = 0, default = 1.0)),
      packages   = "mlr3torchAUM",
      label      = "All-Pairs Squared Hinge Loss"
    )
  )
}
