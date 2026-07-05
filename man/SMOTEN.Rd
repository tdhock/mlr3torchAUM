\name{SMOTEN}
\alias{SMOTEN}
\docType{data}
\title{
  SMOTEN Over-Sampler
}
\description{
  \code{SMOTEN} is an \link[R6:R6Class]{R6} class implementing the variant of
  the Synthetic Minority Over-sampling Technique for Nominal features
  (SMOTEN), ported from the Python \code{imbalanced-learn} library. It is
  meant for data sets whose features are all categorical.
}
\format{
  An \code{\link[R6:R6Class]{R6}} class. Create an instance with
  \code{SMOTEN$new(sampling_strategy = "auto", k_neighbors = 5)}:
  \itemize{
    \item \code{sampling_strategy} A string field defining samples of
    which classes should be synthesized. \code{"auto"}
    over-samples every non-majority class up to
    the size of the majority class. Default; \code{"minority"}
    over-samples only the smallest class.
    Only one class will be selected.
    \item \code{k_neighbors} The hyperparameter \code{k} for knn algorithm.
  }

  After creation, call the \code{fit_resample(X, y)} method:
  \code{X} is a character matrix of categorical features (one row per sample,
  one column per feature) and
  \code{y} is a \code{\link{factor}} of class labels.
  \code{length(y)} should be equal to \code{nrow(X)}.
  It returns a \code{list}
  with two elements: \code{X}, the feature matrix including the original
  and the synthesized ones, and the corresponding \code{y}.

  Unlike \code{SMOTE}, which interpolates continuous features, \code{SMOTEN}
  finds neighbors with the Value Difference Metric (VDM) and synthesizes each
  new sample by taking the most common category (the mode) among a point and
  its neighbors.

  \code{SMOTEN} inherits from
  \code{SMOTE}, \code{BaseSMOTE}, \code{BaseOverSampler} and \code{BaseSampler},
  identical to the class hierarchy of \code{imbalanced-learn}.
}
\author{
Wei Cao
}
\examples{
set.seed(1)
## a purely categorical data set: 8 minority vs 22 majority
X <- cbind(
  color = sample(c("red", "blue", "green"), 30, replace = TRUE),
  taste = sample(c("sweet", "salty"), 30, replace = TRUE)
)
y <- factor(c(rep("min", 8), rep("maj", 22)))
sampler <- mlr3torchAUM::SMOTEN$new(sampling_strategy = "minority", k_neighbors = 3)
res <- sampler$fit_resample(X, y)
table(res$y) # minority over-sampled toward the majority size
}
