torch_loss_weighted_logistic <- function(cost_matrix = NULL) {
  generator <- function(task, ...) {
    if (!("twoclass" %in% task$properties)) {
      stop(sprintf(
        "Weighted logistic loss is only defined for binary classification tasks, but received task '%s'.",
        task$id
      ))
    }
    
    if (!is.null(cost_matrix)) {
      if (!is.matrix(cost_matrix) || nrow(cost_matrix) != 2L || ncol(cost_matrix) != 2L) {
        stop("cost_matrix must be a 2x2 numeric matrix.")
      }
      pos_name <- task$positive
      neg_name <- setdiff(task$class_names, pos_name)
      c_fn <- cost_matrix[neg_name, pos_name]
      c_fp <- cost_matrix[pos_name, neg_name]
      if (is.na(c_fp) || c_fp <= 0) {
        stop("False positive cost in cost_matrix must be greater than zero.")
      }
      pos_w <- as.numeric(c_fn / c_fp)
    } else {
      truth <- task$truth()
      n_pos <- sum(truth == task$positive)
      n_neg <- length(truth) - n_pos
      pos_w <- if (n_pos > 0 && n_neg > 0) (n_neg / n_pos) else 1.0
    }

    torch::nn_bce_with_logits_loss(
      pos_weight = torch::torch_tensor(pos_w, dtype = torch::torch_float())
    )
  }

  mlr3torch::TorchLoss$new(
    torch_loss = generator,
    task_types = "classif",
    id = "weighted_logistic",
    label = "Weighted Logistic Loss",
    packages = c("torch", "mlr3torch"),
    man = "mlr3torchAUM::torch_loss_weighted_logistic"
  )
}
