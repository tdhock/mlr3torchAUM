\name{SMOTE}
\alias{SMOTE}
\docType{data}
\title{
  SMOTE Over-Sampler
}
\description{
  \code{SMOTE} is an \link[R6:R6Class]{R6} class implementing the
  Synthetic Minority Over-sampling Technique (SMOTE), ported from the
  Python \code{imbalanced-learn} library. 
}
\format{
  An \code{\link[R6:R6Class]{R6}} class. Create an instance with
  \code{SMOTE$new(sampling_strategy = "auto", k_neighbors= 5)}:
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
  \code{X} is a matrix of features (one row per sample, 
  one column per feature) and
  \code{y} is a \code{\link{factor}} of class labels.
  \code{length(y)} should be equal to \code{nrow(X)}. 
  It returns a \code{list}
  with two elements: \code{X}, the feature matrix including the original
  and the synthesized ones, and the corresponding \code{y}.

  \code{SMOTE} inherits from the abstract classes
  \code{BaseSMOTE}, \code{BaseOverSampler} and \code{BaseSampler},
  identical to the class hierarchy of \code{imbalanced-learn}.
}
\author{
Wei Cao
}
\examples{
set.seed(1)
## 10 positive vs 40 negative
y <- factor(c(rep("positive", 10), rep("negative", 40)))
X <- matrix(rnorm(100), ncol = 2) # 50 samples, each two features
sampler <- mlr3torchAUM::SMOTE$new(sampling_strategy = "auto", k_neighbors= 5)
res <- sampler$fit_resample(X, y)
nrow(res$X)   # 50 original + 30 synthetic = 80
table(res$y)  # balanced: positive = 40, negative = 40
}
