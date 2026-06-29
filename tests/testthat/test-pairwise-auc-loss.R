library(testthat)
data.table::setDTthreads(1L)

if(torch::torch_is_installed()){

  test_that("nn_pairwise_auc_loss rejects invalid surrogate names", {
    expect_error({
      mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "invalid_loss")
    }, "Unknown surrogate loss")
  })

  test_that("nn_pairwise_auc_loss matches R mathematical formulas for all surrogates", {
    pred_vals <- c(1.0, 0.0, -1.0)
    label_vals <- c(1, 0, 0)
    pred <- torch::torch_tensor(pred_vals)
    label <- torch::torch_tensor(label_vals)
    diffs <- outer(pred_vals[label_vals == 1], pred_vals[label_vals == 0], `-`)
    # 1. Squared
    loss_sq <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "squared", margin = 1.0)
    expect_equal(
      torch::as_array(loss_sq(pred, label)),
      mean((1.0 - diffs)^2)
    )
    # 2. Squared Hinge
    loss_sq_hinge <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "squared_hinge", margin = 1.0)
    expect_equal(
      torch::as_array(loss_sq_hinge(pred, label)),
      mean(pmax(1.0 - diffs, 0)^2)
    )
    # 3. Hinge
    loss_hinge <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "hinge", margin = 1.0)
    expect_equal(
      torch::as_array(loss_hinge(pred, label)),
      mean(pmax(1.0 - diffs, 0))
    )
    # 4. Logistic
    loss_log <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "logistic", scale = 1.0)
    expect_equal(
      torch::as_array(loss_log(pred, label)),
      mean(log1p(exp(-1.0 * diffs)))
    )
    # 5. Barrier Hinge
    loss_barrier <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "barrier_hinge", margin = 1.0, scale = 2.0)
    expect_equal(
      torch::as_array(loss_barrier(pred, label)),
      mean(pmax(-2.0 * (1.0 + diffs) + 1.0, pmax(1.0 - diffs, 2.0 * (diffs - 1.0))))
    )
  })

  test_that("nn_pairwise_auc_loss gradient equals naive autograd gradient", {
    n <- 6
    pred_vals <- c(0.1, -0.5, 0.3, 0.8, -0.2, 0.4)
    label_vals <- c(1L, 0L, 1L, 0L, 1L, 0L)
    pred_fast <- torch::torch_tensor(pred_vals, dtype = torch::torch_double(), requires_grad = TRUE)
    label_t <- torch::torch_tensor(label_vals, dtype = torch::torch_double())
    loss_fn <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "logistic", scale = 1.0)
    loss_fast <- loss_fn(pred_fast, label_t)
    loss_fast$backward()
    pred_slow <- torch::torch_tensor(pred_vals, dtype = torch::torch_double(), requires_grad = TRUE)
    pos_idx <- which(label_vals == 1L)
    neg_idx <- which(label_vals == 0L)
    total_loss <- torch::torch_tensor(0.0, dtype = torch::torch_double())
    for(p in pos_idx) {
      for(n_in in neg_idx) {
        diff <- pred_slow[p] - pred_slow[n_in]
        total_loss <- total_loss + torch::nnf_softplus(-1.0 * diff)
      }
    }
    mean_loss <- total_loss / (length(pos_idx) * length(neg_idx))
    mean_loss$backward()
    expect_equal(
      torch::as_array(pred_fast$grad),
      torch::as_array(pred_slow$grad)
    )
  })

  test_that("nn_pairwise_auc_loss tracks evals, zeros, and all_one_class buffers correctly", {
    loss_fn <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "squared", margin = 1.0)
    pred <- torch::torch_tensor(c(0.5, -0.3))
    label <- torch::torch_tensor(c(1, 0))
    loss <- loss_fn(pred, label)
    expect_equal(torch::as_array(loss_fn$evals), 1L)
    expect_equal(torch::as_array(loss_fn$zeros), 0L)
    expect_equal(torch::as_array(loss_fn$all_one_class), 0L)
    label_same <- torch::torch_tensor(c(1, 1))
    loss_same <- loss_fn(pred, label_same)
    expect_equal(torch::as_array(loss_fn$evals), 2L)
    expect_equal(torch::as_array(loss_fn$all_one_class), 1L)
    expect_equal(torch::as_array(loss_fn$zeros), 1L)
  })

  test_that("nn_pairwise_auc_loss handles all-same-class inputs by returning 0", {
    loss_fn <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "squared", margin = 1.0)
    pred <- torch::torch_tensor(c(0.5, -0.3))
    label_same <- torch::torch_tensor(c(1, 1))
    expect_equal(as.numeric(loss_fn(pred, label_same)), 0)
  })

}
