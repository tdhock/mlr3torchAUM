\name{SMOTENC}
\alias{SMOTENC}
\docType{data}
\title{
  SMOTENC Over-Sampler
}
\description{
  \code{SMOTENC} is an \link[R6:R6Class]{R6} class implementing the variant of
  the Synthetic Minority Over-sampling Technique for Nominal and Continuous
  features (SMOTE-NC), ported from the Python \code{imbalanced-learn} library.
  It is meant for data sets with a mix of continuous and categorical features.
}
\format{
  An \code{\link[R6:R6Class]{R6}} class. Create an instance with
  \code{SMOTENC$new(sampling_strategy = "auto", k_neighbors = 5, categorical_features = "auto")}:
  \itemize{
    \item \code{sampling_strategy} A string field defining samples of
    which classes should be synthesized. \code{"auto"}
    over-samples every non-majority class up to
    the size of the majority class. Default; \code{"minority"}
    over-samples only the smallest class.
    Only one class will be selected.
    \item \code{k_neighbors} The hyperparameter \code{k} for knn algorithm.
    \item \code{categorical_features} Which columns are categorical: a vector of
    column indices or names, or \code{"auto"} (default) to detect the character
    and factor columns automatically.
  }

  After creation, call the \code{fit_resample(X, y)} method:
  \code{X} is a \code{\link{data.frame}} with a mix of continuous (numeric) and
  categorical (character or factor) columns (one row per sample, one column per
  feature) and
  \code{y} is a \code{\link{factor}} of class labels.
  \code{length(y)} should be equal to \code{nrow(X)}.
  It returns a \code{list}
  with two elements: \code{X}, the feature matrix including the original
  and the synthesized ones, and the corresponding \code{y}.

  \code{SMOTENC} synthesizes each new sample by interpolating the continuous
  features (as in \code{SMOTE}) and taking the most common category (the mode)
  among the neighbors for the categorical features (as in \code{SMOTEN}).
  Neighbors are found with a distance that combines the continuous Euclidean
  distance with a penalty, for every mismatching categorical feature, equal to
  the median of the per-class continuous feature standard deviations.

  \code{SMOTENC} inherits from
  \code{SMOTE}, \code{BaseSMOTE}, \code{BaseOverSampler} and \code{BaseSampler},
  identical to the class hierarchy of \code{imbalanced-learn}.
}
\author{
Wei Cao
}
\examples{
set.seed(1)
## a mixed data set: 8 minority vs 22 majority
X <- data.frame(
  height = rnorm(30, 170, 10),
  color = sample(c("red", "blue", "green"), 30, replace = TRUE),
  taste = sample(c("sweet", "salty"), 30, replace = TRUE),
  stringsAsFactors = FALSE
)
y <- factor(c(rep("min", 8), rep("maj", 22)))
sampler <- mlr3torchAUM::SMOTENC$new(
  sampling_strategy = "minority", k_neighbors = 3, categorical_features = c(2, 3)
)
res <- sampler$fit_resample(X, y)
table(res$y) # minority over-sampled toward the majority size
}
