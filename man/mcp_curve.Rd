\name{mcp_curve}
\alias{mcp_curve}
\alias{mcp_score}
\alias{imcp_curve}
\alias{imcp_score}
\title{
  (Imbalanced) Multiclass Classification Performance (MCP/IMCP) curves and areas
}
\description{
  The MCP curve summarizes multiclass classification performance using
  the Hellinger distance between the predicted class probabilities and
  the one-hot encoded true labels. The imbalanced variant (IMCP) weights
  each sample by the inverse frequency of its class, so the curve is not
  dominated by majority classes. The \code{*_score} functions return the
  area under the corresponding curve, computed by the trapezoidal rule.
  All functions operate on \code{torch} tensors and are differentiable
  with respect to \code{pred_tensor} (gradients flow through the per-sample
  scores, with the sort order held fixed).
}
\usage{
mcp_curve(pred_tensor, label_tensor, abs_tolerance = 1e-6)
mcp_score(pred_tensor, label_tensor, abs_tolerance = 1e-6)
imcp_curve(pred_tensor, label_tensor, abs_tolerance = 1e-6)
imcp_score(pred_tensor, label_tensor, abs_tolerance = 1e-6)
}
\arguments{
  \item{pred_tensor}{A \code{torch} tensor of predicted class
    probabilities, of shape \code{(n_samples, n_classes)}. Each row must
    sum to 1. Columns correspond to classes \code{1} to \code{n_classes}.}
  \item{label_tensor}{A \code{torch} long tensor of true class labels,
    1-based (values \code{1} to \code{n_classes}), of length
    \code{n_samples}. Encoding the labels (factor to integer codes) is the
    caller's responsibility; see the mlr3 measures \code{classif.mcp} /
    \code{classif.imcp} for the typical entry point.}
  \item{abs_tolerance}{Absolute tolerance used when checking that each row
    of \code{pred_tensor} sums to 1. The default \code{1e-6} matches the
    decimal resolution of 32-bit floats.}
}
\value{
  \code{mcp_curve} and \code{imcp_curve} return a list with two \code{torch}
  tensors of equal length, \code{curve_x} and \code{curve_y}, giving the
  coordinates of the (I)MCP curve. \code{mcp_score} and \code{imcp_score}
  return a scalar \code{torch} tensor: the area under the corresponding
  curve. Convert a tensor to a plain R number with \code{torch::as_array()}
  or its \code{$item()} method.
}
\references{
  Aguilar-Ruiz, J. S. and Michalak, M. (2024) Classification performance
  assessment for imbalanced multiclass data. Scientific Reports 14:10759.
  \doi{10.1038/s41598-024-61365-z}
}
\author{
Wei Cao
}
\examples{

if (torch::torch_is_installed()) {
  ## 4 samples, 3 classes; labels are 1-based long.
  pred <- torch::torch_tensor(rbind(
    c(0.7, 0.2, 0.1),
    c(0.3, 0.4, 0.3),
    c(0.2, 0.6, 0.2),
    c(0.1, 0.3, 0.6)))
  label <- torch::torch_tensor(c(1L, 1L, 2L, 3L), dtype = torch::torch_long())

  ## MCP curve coordinates and area under the curve.
  mcp_curve(pred, label)
  torch::as_array(mcp_score(pred, label))

  ## Imbalanced variant weights rare classes more heavily.
  torch::as_array(imcp_score(pred, label))
}

}
