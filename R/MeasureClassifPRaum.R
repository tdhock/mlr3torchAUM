#' Custom mlr3 Measure: PR AUM
#'
#' An [mlr3::MeasureClassif] implementation that evaluates binary classifiers
#' using the PR AUM metric (Area Under Minimum(FDR,FNR)).
#'
#' @format [R6::R6Class] object.
#'
#' @details
#' This measure is designed for imbalanced binary classification. It treats
#' labels as `-1` (negative class) and `+1` (positive class), converts
#' predictions into torch tensors, and computes the AUM score via
#' [PR_AUM()]. The measure is minimized, with values closer to 0
#' indicating better performance.
#'
#' The `initialize()` method sets up the measure with default properties.
#' Users should call `MeasureClassifPRAUM$new()` rather than `initialize()` directly.
#'
#' @seealso [PR_AUM()], [PR_curve()]
#' @references
#' Original implementation ideas adapted from:
#' \itemize{
#'   \item \url{https://github.com/OGuenoun/R-AUM_Multiclass/blob/main/AUM_comparison.r}
#'   \item \url{https://github.com/tdhock/imbalanced-paper/blob/main/2025-07-30-conv-linear-prop0.01.R}
#' }
#' @export

MeasureClassifPRAUM = R6::R6Class(
  "MeasureClassifPRAUM",
  inherit = mlr3::MeasureClassif,
  public = list(
    initialize = function() { 
      super$initialize(
        id = "classif.praum",
        label = "Precision-Recall Area Under the Minimum",
        packages = "torch",
        properties = character(),
        task_properties = "twoclass",
        predict_type = "prob",
        range = c(0, Inf),
        minimize = TRUE
      )
    }
  ),
  private = list(
    .score = function(prediction, ...) {
      pred_tensor <- torch::torch_tensor(prediction$prob[, 1])
      label_tensor <- torch::torch_tensor(prediction$truth)
      loss_tensor <- PR_AUM(pred_tensor, label_tensor)
      torch::as_array(loss_tensor)
    }
  )
)
