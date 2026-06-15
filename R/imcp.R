trapz <- function(x_tensor, y_tensor) {
  length_y <- y_tensor$size(1)
  sum((y_tensor[2:length_y] + y_tensor[1:(length_y - 1)]) *
    torch::torch_diff(x_tensor) / 2)
}

get_y_values <- function(
  pred_tensor, y_true_one_hot_tensor, label_tensor
) {
  hellinger_distance <- torch::torch_sqrt(
    ((y_true_one_hot_tensor - torch::torch_sqrt(pred_tensor))^2)$sum(2)
  ) / sqrt(2)
  curve_y <- 1 - hellinger_distance
  ord <- order(
    torch::as_array(curve_y),
    torch::as_array(label_tensor)
  ) # TODO(wei) research on
  return(list(curve_y = curve_y[ord], sort_indices = ord))
}

prepare_curve <- function(pred_tensor, label_tensor, abs_tolerance = 1e-6) {
  n_samples <- pred_tensor$size(1)
  n_classes <- pred_tensor$size(2)
  if (n_samples != label_tensor$size(1)) stop("sample numbers don't match")
  if (any(abs(torch::as_array(pred_tensor$sum(dim = 2) - 1)) > abs_tolerance)) {
    stop("Not all rows' probabilities sum to 1")
  }
  y_true_one_hot_tensor <- torch::nnf_one_hot(label_tensor, num_classes = n_classes)
  y_values <- get_y_values(pred_tensor, y_true_one_hot_tensor, label_tensor)
  return(list(
    n_samples = n_samples, n_classes = n_classes,
    y_true_one_hot_tensor = y_true_one_hot_tensor,
    curve_y = y_values$curve_y, sort_indices = y_values$sort_indices
  ))
}

mcp_curve <- function(pred_tensor, label_tensor, abs_tolerance = 1e-6) {
  prepared <- prepare_curve(pred_tensor, label_tensor, abs_tolerance)
  n_samples <- prepared$n_samples
  device <- pred_tensor$device
  return(list(
    curve_x = torch::torch_tensor(
      0:(n_samples - 1) / (n_samples - 1),
      device = device
    ),
    curve_y = prepared$curve_y
  ))
}

mcp_score <- function(pred_tensor, label_tensor, abs_tolerance = 1e-6) {
  cur <- mcp_curve(pred_tensor, label_tensor, abs_tolerance)
  return(trapz(cur$curve_x, cur$curve_y))
}

get_class_widths <- function(y_true_one_hot_tensor, n_classes) {
  counts <- y_true_one_hot_tensor$sum(dim = 1)
  return(1 / (n_classes * counts))
}

imcp_curve <- function(pred_tensor, label_tensor, abs_tolerance = 1e-6) {
  prepared <- prepare_curve(pred_tensor, label_tensor, abs_tolerance)
  device <- pred_tensor$device
  class_widths <- get_class_widths(prepared$y_true_one_hot_tensor, prepared$n_classes)
  sample_widths <- class_widths[label_tensor]
  curve_x <- sample_widths[prepared$sort_indices]
  curve_x <- curve_x$cumsum(dim = 1) - curve_x / 2
  curve_x <- torch::torch_cat(
    list(
      torch::torch_zeros(1, device = device),
      curve_x,
      torch::torch_ones(1, device = device)
    )
  )
  cy <- prepared$curve_y
  n <- cy$size(1)
  curve_y <- torch::torch_cat(list(cy[1:1], cy, cy[n:n]))
  return(list(curve_x = curve_x, curve_y = curve_y))
}

imcp_score <- function(pred_tensor, label_tensor, abs_tolerance = 1e-6) {
  cur <- imcp_curve(pred_tensor, label_tensor, abs_tolerance)
  return(trapz(cur$curve_x, cur$curve_y))
}

MeasureClassifIMCP <- R6Class(
  "IMCP",
  inherit = MeasureClassif,
  public = list(
    initialize = function() {
      super$initialize(
        id = "classif.imcp",
        label = "Area Under the Imbalanced Multiclass Classification Performance curve",
        packages = "torch",
        properties = character(),
        task_properties = character(), # multiclass
        predict_type = "prob",
        range = c(0, 1),
        minimize = FALSE
      )
    }
  ),
  private = list(
    .score = function(prediction, ...) {
      pred_tensor <- torch::torch_tensor(prediction$prob)
      label_tensor <- torch::torch_tensor(as.integer(prediction$truth),
        dtype = torch::torch_long()
      )
      torch::as_array(imcp_score(pred_tensor, label_tensor))
    }
  )
)

nn_IMCP_loss <- torch::nn_module(
  c("nn_IMCP_loss", "nn_loss"),
  forward = function(pred_tensor, label_tensor) {
    1 - imcp_score(torch::nnf_softmax(pred_tensor, dim = 2), label_tensor)
  }
)
MeasureClassifMCP <- R6Class(
  "MCP",
  inherit = MeasureClassif,
  public = list(
    initialize = function() {
      super$initialize(
        id = "classif.mcp",
        label = "Area Under the Multiclass Classification Performance curve",
        packages = "torch",
        properties = character(),
        task_properties = character(), # multiclass
        predict_type = "prob",
        range = c(0, 1),
        minimize = FALSE
      )
    }
  ),
  private = list(
    .score = function(prediction, ...) {
      pred_tensor <- torch::torch_tensor(prediction$prob)
      label_tensor <- torch::torch_tensor(as.integer(prediction$truth),
        dtype = torch::torch_long()
      )
      torch::as_array(mcp_score(pred_tensor, label_tensor))
    }
  )
)
