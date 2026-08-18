\name{torch_loss_weighted_logistic}
\alias{torch_loss_weighted_logistic}
\alias{weighted_logistic}
\title{
  Weighted Binary Logistic Loss
}
\description{
  Creates a weighted logistic loss for binary classification in \code{mlr3torch}.
  It automatically weights the positive class by the class frequency ratio in the training data,
  or by the ratio of False Negative to False Positive costs from a user-supplied cost matrix.
}
\usage{
torch_loss_weighted_logistic(cost_matrix = NULL)
}
\arguments{
  \item{cost_matrix}{Optional 2x2 numeric matrix with row names (predicted class)
    and column names (true class). If supplied, the positive class weight is computed
    as the cost ratio \eqn{c_{FN} / c_{FP}}. If \code{NULL} (default), the weight is automatically
    computed from the training task as \eqn{N_{neg} / N_{pos}}.}
}
\details{
  This loss function is registered in the \code{mlr3torch} loss dictionary
  under the key \code{"weighted_logistic"}. It can be instantiated via
  \code{mlr3torch::t_loss("weighted_logistic")} or directly using \code{torch_loss_weighted_logistic()}.
  When providing a \code{cost_matrix}, it must be a 2x2 matrix with row and column names matching the task class names:
  \itemize{
    \item \strong{Rows}: Predicted class.
    \item \strong{Columns}: True class.
  }
  The underlying loss is computed using \code{torch::nn_bce_with_logits_loss(pos_weight = pos_w)}.
}
\value{
  A \code{mlr3torch::TorchLoss} object, suitable for use as the \code{loss} parameter
  of a \code{TorchLearner}.
}
\references{
  Elkan, Charles (2001). The Foundations of Cost-Sensitive Learning.
  Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence (IJCAI).

  Bischl, B., Sonabend, R., Kotthoff, L., & Lang, M. (Eds.). (2024).
  Applied Machine Learning Using mlr3 in R.
  CRC Press. \url{https://mlr3book.mlr-org.com}
}
\examples{
if (torch::torch_is_installed() && requireNamespace("mlr3", quietly = TRUE)) {
  # 1. Automatic frequency weighting
  loss_auto <- mlr3torchAUM::torch_loss_weighted_logistic()
  # 2. Custom cost matrix weighting (rows = predicted, cols = truth)
  cm <- matrix(c(0, 10, 2, 0), nrow = 2, dimnames = list(
    predicted = c("M", "R"),
    truth     = c("M", "R")
  ))
  loss_cm <- mlr3torchAUM::torch_loss_weighted_logistic(cost_matrix = cm)
}
}
