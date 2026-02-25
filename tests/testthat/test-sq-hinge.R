## tests/testthat/test-loss-squared-hinge.R
##
## Tests for all_pairs_squared_hinge_loss and all_pairs_squared_hinge_loss_vec.
## Run with:  testthat::test_file("tests/testthat/test-loss-squared-hinge.R")
## or:        devtools::test()

library(testthat)
library(torch)

## Note: when running via devtools::test() or R CMD check, package functions
## are loaded automatically. The source() below is only for interactive use.
if (!exists("all_pairs_squared_hinge_loss")) {
  source(file.path(rprojroot::find_root(rprojroot::is_r_package),
                   "R", "sq_hinge_loss.R"))
}

## ── helpers ──────────────────────────────────────────────────────────────────

## Compute loss value as a plain R scalar
loss_val <- function(fn, pred, label, margin = 1) {
  p   <- torch_tensor(as.numeric(pred), requires_grad = FALSE)
  res <- fn(p, label = label, margin = margin)
  as.numeric(res)
}

## Compute gradient vector via autograd
loss_grad <- function(fn, pred, label, margin = 1) {
  p <- torch_tensor(as.numeric(pred), requires_grad = TRUE)
  l <- fn(p, label = label, margin = margin)
  l$backward()
  as.numeric(p$grad)
}

## Finite-difference numerical gradient (central differences)
numerical_grad <- function(fn, pred, label, margin = 1, eps = 1e-4) {
  sapply(seq_along(pred), function(i) {
    pred_hi <- pred; pred_hi[i] <- pred[i] + eps
    pred_lo <- pred; pred_lo[i] <- pred[i] - eps
    (loss_val(fn, pred_hi, label, margin) -
        loss_val(fn, pred_lo, label, margin)) / (2 * eps)
  })
}

## ── Test suite ────────────────────────────────────────────────────────────────

test_that("loss is zero when all positive predictions exceed all negative + margin", {
  ## y_hat_pos - y_hat_neg = 3 - 0 = 3 > margin=1 → hinge = max(0, 1-3)^2 = 0
  pred  <- c(3.0, 0.0)
  label <- c(1L, -1L)
  expect_equal(loss_val(all_pairs_squared_hinge_loss,     pred, label), 0.0)
  expect_equal(loss_val(all_pairs_squared_hinge_loss_vec, pred, label), 0.0)
})

test_that("loss equals naive implementation on simple two-pair case", {
  ## One positive (y=2), one negative (y=-1):
  ## loss = max(0, 1 - (2 - (-1)))^2 = max(0, 1-3)^2 = 0
  pred  <- c(2.0, -1.0)
  label <- c(1L, -1L)
  naive  <- loss_val(all_pairs_squared_hinge_loss_naive, pred, label)
  fast   <- loss_val(all_pairs_squared_hinge_loss,       pred, label)
  vecfn  <- loss_val(all_pairs_squared_hinge_loss_vec,   pred, label)
  expect_equal(fast,  naive, tolerance = 1e-5)
  expect_equal(vecfn, naive, tolerance = 1e-5)
})

test_that("loss equals naive on a case with non-zero loss", {
  ## pos=0, neg=0 → diff = 0, hinge = max(0, 1-0)^2 = 1
  pred  <- c(0.0, 0.0)
  label <- c(1L, -1L)
  naive  <- loss_val(all_pairs_squared_hinge_loss_naive, pred, label)
  fast   <- loss_val(all_pairs_squared_hinge_loss,       pred, label)
  vecfn  <- loss_val(all_pairs_squared_hinge_loss_vec,   pred, label)
  expect_equal(fast,  1.0, tolerance = 1e-5)
  expect_equal(vecfn, 1.0, tolerance = 1e-5)
  expect_equal(fast,  naive, tolerance = 1e-5)
  expect_equal(vecfn, naive, tolerance = 1e-5)
})

test_that("loss equals naive on random predictions", {
  set.seed(42)
  for (trial in seq_len(10)) {
    n     <- sample(4:12, 1)
    pred  <- rnorm(n)
    ## ensure at least one positive and one negative
    label <- c(1L, -1L, sample(c(-1L, 1L), n - 2, replace = TRUE))
    naive  <- loss_val(all_pairs_squared_hinge_loss_naive, pred, label)
    fast   <- loss_val(all_pairs_squared_hinge_loss,       pred, label)
    vecfn  <- loss_val(all_pairs_squared_hinge_loss_vec,   pred, label)
    expect_equal(fast,  naive, tolerance = 1e-4,
                 label = sprintf("trial %d fast vs naive", trial))
    expect_equal(vecfn, naive, tolerance = 1e-4,
                 label = sprintf("trial %d vec vs naive", trial))
  }
})

test_that("loss equals naive with non-default margin", {
  set.seed(7)
  pred  <- rnorm(8)
  label <- c(1L, 1L, -1L, -1L, 1L, -1L, 1L, -1L)
  for (m in c(0, 0.5, 2, 5)) {
    naive  <- loss_val(all_pairs_squared_hinge_loss_naive, pred, label, m)
    fast   <- loss_val(all_pairs_squared_hinge_loss,       pred, label, m)
    vecfn  <- loss_val(all_pairs_squared_hinge_loss_vec,   pred, label, m)
    expect_equal(fast,  naive, tolerance = 1e-4,
                 label = sprintf("margin=%g fast", m))
    expect_equal(vecfn, naive, tolerance = 1e-4,
                 label = sprintf("margin=%g vec", m))
  }
})

test_that("gradient of loop version matches numerical gradient", {
  set.seed(1)
  pred  <- rnorm(6)
  label <- c(1L, -1L, 1L, -1L, 1L, -1L)
  g_auto <- loss_grad(all_pairs_squared_hinge_loss, pred, label)
  g_num  <- numerical_grad(all_pairs_squared_hinge_loss, pred, label)
  expect_equal(g_auto, g_num, tolerance = 1e-3)
})

test_that("gradient of vectorised version matches numerical gradient", {
  set.seed(2)
  pred  <- rnorm(8)
  label <- c(1L, 1L, -1L, -1L, 1L, -1L, -1L, 1L)
  g_auto <- loss_grad(all_pairs_squared_hinge_loss_vec, pred, label)
  g_num  <- numerical_grad(all_pairs_squared_hinge_loss_vec, pred, label)
  expect_equal(g_auto, g_num, tolerance = 1e-3)
})

test_that("gradient of loop and vectorised versions agree", {
  set.seed(3)
  for (trial in seq_len(10)) {
    n     <- sample(4:12, 1)
    pred  <- rnorm(n)
    label <- c(1L, -1L, sample(c(-1L, 1L), n - 2, replace = TRUE))
    g_loop <- loss_grad(all_pairs_squared_hinge_loss,     pred, label)
    g_vec  <- loss_grad(all_pairs_squared_hinge_loss_vec, pred, label)
    expect_equal(g_loop, g_vec, tolerance = 1e-4,
                 label = sprintf("trial %d gradient agreement", trial))
  }
})

test_that("loss is zero when there are no positive examples", {
  pred  <- c(1.0, 2.0, 3.0)
  label <- c(-1L, -1L, -1L)
  expect_equal(loss_val(all_pairs_squared_hinge_loss,     pred, label), 0.0)
  expect_equal(loss_val(all_pairs_squared_hinge_loss_vec, pred, label), 0.0)
})

test_that("loss is zero when there are no negative examples", {
  pred  <- c(1.0, 2.0, 3.0)
  label <- c(1L, 1L, 1L)
  expect_equal(loss_val(all_pairs_squared_hinge_loss,     pred, label), 0.0)
  expect_equal(loss_val(all_pairs_squared_hinge_loss_vec, pred, label), 0.0)
})

test_that("loss is non-negative (convexity check)", {
  set.seed(99)
  for (trial in seq_len(20)) {
    n     <- sample(4:15, 1)
    pred  <- rnorm(n) * 3
    label <- c(1L, -1L, sample(c(-1L, 1L), n - 2, replace = TRUE))
    v <- loss_val(all_pairs_squared_hinge_loss_vec, pred, label)
    expect_gte(v, 0.0, label = sprintf("trial %d non-negativity", trial))
  }
})

test_that("loss decreases as positive predictions increase above negatives", {
  label <- c(1L, -1L)
  losses <- sapply(seq(-3, 3, by = 0.5), function(pos_pred) {
    loss_val(all_pairs_squared_hinge_loss_vec, c(pos_pred, 0.0), label)
  })
  ## Losses should be non-increasing as pos_pred increases
  expect_true(all(diff(losses) <= 1e-6))
})

test_that("manual calculation: two positives, one negative, margin=1", {
  ## pos1=1, pos2=2, neg=0, margin=1
  ## pair (pos1, neg): max(0, 1-(1-0))^2 = max(0,0)^2 = 0
  ## pair (pos2, neg): max(0, 1-(2-0))^2 = max(0,-1)^2 = 0
  pred  <- c(1.0, 2.0, 0.0)
  label <- c(1L, 1L, -1L)
  expect_equal(loss_val(all_pairs_squared_hinge_loss_vec, pred, label), 0.0)
  expect_equal(loss_val(all_pairs_squared_hinge_loss,     pred, label), 0.0)
  
  ## pos1=-1, pos2=0, neg=1, margin=1
  ## pair (pos1, neg): max(0, 1-(-1-1))^2 = max(0,3)^2 = 9
  ## pair (pos2, neg): max(0, 1-(0-1))^2  = max(0,2)^2 = 4
  ## total = 13
  pred2  <- c(-1.0, 0.0, 1.0)
  label2 <- c(1L, 1L, -1L)
  naive2 <- loss_val(all_pairs_squared_hinge_loss_naive, pred2, label2)
  fast2  <- loss_val(all_pairs_squared_hinge_loss,       pred2, label2)
  vec2   <- loss_val(all_pairs_squared_hinge_loss_vec,   pred2, label2)
  expect_equal(naive2, 13.0, tolerance = 1e-5)
  expect_equal(fast2,  naive2, tolerance = 1e-5)
  expect_equal(vec2,   naive2, tolerance = 1e-5)
})

test_that("output is a scalar torch tensor", {
  pred  <- torch_tensor(c(1.0, -1.0))
  label <- c(1L, -1L)
  out_loop <- all_pairs_squared_hinge_loss(pred, label)
  out_vec  <- all_pairs_squared_hinge_loss_vec(pred, label = label)
  expect_true(inherits(out_loop, "torch_tensor"))
  expect_true(inherits(out_vec,  "torch_tensor"))
  expect_equal(length(out_loop$shape), 0L)   # scalar: shape is integer(0)
  expect_equal(length(out_vec$shape),  0L)
})

test_that("gradient is zero at a perfectly ranked solution", {
  ## When every positive is >> every negative by more than margin,
  ## the hinge is 0 everywhere, so gradient should be 0.
  pred  <- c(5.0, 6.0, -5.0, -6.0)
  label <- c(1L, 1L, -1L, -1L)
  g <- loss_grad(all_pairs_squared_hinge_loss_vec, pred, label)
  expect_equal(g, rep(0.0, 4), tolerance = 1e-6)
})

message("All squared hinge loss tests passed.")
