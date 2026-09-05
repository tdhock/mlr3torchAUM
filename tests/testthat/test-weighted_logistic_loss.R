library(testthat)
data.table::setDTthreads(1L)

if (torch::torch_is_installed()) {

  test_that("torch_loss_weighted_logistic rejects invalid arguments", {
    loss_gen <- mlr3torchAUM::torch_loss_weighted_logistic()
    task_multi <- mlr3::tsk("iris")
    expect_error({
      loss_gen$generate(task_multi)
    }, "currently only implemented for binary classification tasks")
    loss_bad_cm1 <- mlr3torchAUM::torch_loss_weighted_logistic(cost_matrix = c(1, 2, 3, 4))
    task_bin <- mlr3::tsk("sonar")
    expect_error({
      loss_bad_cm1$generate(task_bin)
    }, "cost_matrix must be a 2x2 numeric matrix")
    bad_cm <- matrix(c(0, 10, 0, 0), nrow = 2, dimnames = list(c("M", "R"), c("M", "R")))
    loss_bad_cm2 <- mlr3torchAUM::torch_loss_weighted_logistic(cost_matrix = bad_cm)
    expect_error({
      loss_bad_cm2$generate(task_bin)
    }, "False positive cost in cost_matrix must be greater than zero")
  })

  test_that("torch_loss_weighted_logistic parses mlr3 cost matrix correctly", {
    # c_fn: pred = R, truth = M (10)
    # c_fp: pred = M, truth = R (2)
    cm <- matrix(c(0, 10, 2, 0), nrow = 2, dimnames = list(
      predicted = c("M", "R"),
      truth = c("M", "R")
    ))
    task <- mlr3::tsk("sonar")
    loss <- mlr3torchAUM::torch_loss_weighted_logistic(cost_matrix = cm)
    loss_fn <- loss$generate(task)
    expect_equal(as.numeric(loss_fn$pos_weight), 5.0)
  })

  test_that("torch_loss_weighted_logistic auto-computes weights and matches manual BCE", {
    data <- data.frame(
      feat = rnorm(10),
      target = factor(c(rep("pos", 2), rep("neg", 8)), levels = c("pos", "neg"))
    )
    task <- mlr3::as_task_classif(data, target = "target", positive = "pos")
    loss <- mlr3torchAUM::torch_loss_weighted_logistic()
    loss_fn <- loss$generate(task)
    expect_equal(as.numeric(loss_fn$pos_weight), 4.0)
    logits <- c(0.5, -1.0, 2.0)
    y_true <- c(1, 0, 1)
    sigmoid <- function(x) 1 / (1 + exp(-x))
    p <- sigmoid(logits)
    manual_loss <- mean(-4.0 * y_true * log(p) - (1 - y_true) * log(1 - p))
    torch_loss_val <- as.numeric(loss_fn(
      torch::torch_tensor(logits, dtype = torch::torch_double()),
      torch::torch_tensor(y_true, dtype = torch::torch_double())
    ))
    expect_equal(torch_loss_val, manual_loss)
  })
}
