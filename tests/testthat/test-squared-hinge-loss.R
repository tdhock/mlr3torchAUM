library(testthat)
data.table::setDTthreads(1L)

## Naive O(n^2) reference implementing eq. (2) from arXiv:2302.11062.
naive_squared_hinge_loss <- function(pred, label, margin=1){
  pos <- which(label == 1)
  neg <- which(label != 1)
  if(length(pos) == 0 || length(neg) == 0) return(0)
  total <- 0
  for(j in pos){
    for(k in neg){
      total <- total + max(margin - (pred[j] - pred[k]), 0)^2
    }
  }
  total
}

if(torch::torch_is_installed()){

  test_that("SquaredHingeLoss rejects invalid margin", {
    pred <- torch::torch_tensor(c(1.0, -1.0))
    label <- torch::torch_tensor(c(1, -1))
    expect_error(
      mlr3torchAUM::SquaredHingeLoss(pred, label, margin=-1),
      "non-negative")
  })

  test_that("SquaredHingeLoss matches naive O(n^2) implementation", {
    n <- 10
    pred <- rnorm(n)
    label <- rep(c(1L, -1L), length.out=n)     
    fast_loss <- torch::as_array(
      mlr3torchAUM::SquaredHingeLoss(
        torch::torch_tensor(pred, dtype=torch::torch_double()),
        torch::torch_tensor(label)
      )
    )
    slow_loss <- naive_squared_hinge_loss(pred, label)
    expect_equal(fast_loss, slow_loss)
  })

  test_that("SquaredHingeLoss gradient equals naive autograd gradient", {
    n <- 8
    pred_vals <- rnorm(n)
    label_vals <- rep(c(1L, -1L), length.out=n)     
    label_t <- torch::torch_tensor(label_vals)        
    pred_fast <- torch::torch_tensor(
      pred_vals, dtype=torch::torch_double(), requires_grad=TRUE)
    mlr3torchAUM::SquaredHingeLoss(pred_fast, label_t)$backward()
    pred_slow <- torch::torch_tensor(
      pred_vals, dtype=torch::torch_double(), requires_grad=TRUE)
    pos <- which(label_vals == 1L) # Now strictly comparing integer to integer
    neg <- which(label_vals != 1L) 
    total <- torch::torch_tensor(0, dtype=torch::torch_double())
    margin_t <- torch::torch_tensor(1.0, dtype=torch::torch_double())    
    for(j in pos) {
      for(k in neg) {
        hinge <- torch::torch_clamp(margin_t - (pred_slow[j] - pred_slow[k]), min=0)
        total <- total + hinge^2
      }
    }
    total$backward()
    expect_equal(torch::as_array(pred_fast$grad), torch::as_array(pred_slow$grad))
  })

  test_that("nn_squared_hinge_loss module tracks buffers", {
    loss_fn <- mlr3torchAUM::nn_squared_hinge_loss(margin=1)
    pred <- torch::torch_tensor(c(0.5, -0.3))
    label <- torch::torch_tensor(c(1, -1))
    loss <- loss_fn(pred, label)
    expect_equal(torch::as_array(loss_fn$evals), 1L)
    ## Call with all same class to trigger all_one_class and zeros counters.
    label_same <- torch::torch_tensor(c(1, 1))
    loss2 <- loss_fn(pred, label_same)
    expect_equal(torch::as_array(loss_fn$evals), 2L)
    expect_equal(torch::as_array(loss_fn$all_one_class), 1L)
    expect_equal(torch::as_array(loss_fn$zeros), 1L)
  })
  
  test_that("sq_hinge_loglinear dictionary retrieval works", {
    skip_if_not_installed("mlr3torch")
    expect_true("sq_hinge_loglinear" %in% mlr3torch::mlr3torch_losses$keys())
    expect_true("TorchLoss" %in% class(mlr3torch::t_loss("sq_hinge_loglinear")))
  })

}
