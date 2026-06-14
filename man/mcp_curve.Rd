\name{mcp_curve}
\alias{mcp_curve}
\alias{mcp_score}
\alias{imcp_curve}
\alias{imcp_score}
\title{
  Multiclass Classification Performance (MCP) curves and areas
}
\description{
  The MCP curve summarizes multiclass classification performance using
  the Hellinger distance between the predicted class probabilities and
  the one-hot encoded true labels. The imbalanced variant (IMCP) weights
  each sample by the inverse frequency of its class, so the curve is not
  dominated by majority classes. The \code{*_score} functions return the
  area under the corresponding curve, computed by the trapezoidal rule.
}
\usage{
mcp_curve(y_true, y_score, labels = NULL, abs_tolerance = 1e-8)
mcp_score(y_true, y_score, labels = NULL, abs_tolerance = 1e-8)
imcp_curve(y_true, y_score, labels = NULL, abs_tolerance = 1e-8)
imcp_score(y_true, y_score, labels = NULL, abs_tolerance = 1e-8)
}
\arguments{
  \item{y_true}{Vector of true class labels, of length equal to the
    number of samples. May be numeric or character.}
  \item{y_score}{Numeric matrix of predicted class probabilities, with
    one row per sample and one column per class. Each row must sum to 1.
    Columns correspond to the class labels in increasing (numeric or
    lexicographic) order.}
  \item{labels}{Optional vector of all class labels mapped to the columns
    of \code{y_score}. Required when some class represented by a column of
    \code{y_score} does not appear in \code{y_true}. Must have length
    equal to the number of columns of \code{y_score} and the same type as
    \code{y_true}.}
  \item{abs_tolerance}{Absolute tolerance used when checking that each row
    of \code{y_score} sums to 1.}
}
\value{
  \code{mcp_curve} and \code{imcp_curve} return a list with two numeric
  vectors of equal length, \code{curve_x} and \code{curve_y}, giving the
  coordinates of the (I)MCP curve. \code{mcp_score} and \code{imcp_score}
  return a single number: the area under the corresponding curve.
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

## 4 samples, 3 classes.
y_true <- c(0, 0, 1, 2)
y_score <- rbind(
  c(0.7, 0.2, 0.1),
  c(0.3, 0.4, 0.3),
  c(0.2, 0.6, 0.2),
  c(0.1, 0.3, 0.6))

## MCP curve coordinates and area under the curve.
mcp_curve(y_true, y_score)
mcp_score(y_true, y_score)

## Imbalanced variant weights rare classes more heavily.
imcp_curve(y_true, y_score)
imcp_score(y_true, y_score)

}
