\name{MeasureClassifIMCP}
\alias{MeasureClassifIMCP}
\alias{MeasureClassifMCP}
\docType{data}
\title{
  mlr3 measures for (Imbalanced) Multiclass Classification Performance
}
\description{
  Area under the (I)MCP curve, exposed as mlr3 classification measures with
  ids \code{"classif.mcp"} and \code{"classif.imcp"}. Higher is better
  (\code{minimize = FALSE}), range \code{[0, 1]}, and they require
  probability predictions (\code{predict_type = "prob"}). Retrieve them with
  \code{\link[mlr3]{msr}()} and use them like any mlr3 measure, e.g. as the
  \code{measures_train} / \code{measures_valid} of a torch Learner.
}
\format{R6 Class inheriting from \code{\link[mlr3]{MeasureClassif}}.}
\examples{

if (requireNamespace("mlr3") && torch::torch_is_installed()) {
  truth <- factor(c("a", "a", "b", "c"), levels = c("a", "b", "c"))
  prob <- matrix(
    c(0.7, 0.3, 0.2, 0.1,
      0.2, 0.4, 0.6, 0.3,
      0.1, 0.3, 0.2, 0.6),
    ncol = 3, dimnames = list(NULL, c("a", "b", "c")))
  p <- mlr3::PredictionClassif$new(row_ids = 1:4, truth = truth, prob = prob)

  p$score(mlr3::msr("classif.imcp"))
  p$score(mlr3::msr("classif.mcp"))
}

}
