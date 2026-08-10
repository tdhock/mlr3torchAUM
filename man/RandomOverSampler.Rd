\name{RandomOverSampler}
\alias{RandomOverSampler}
\docType{data}
\title{
  Random Over-Sampler
}
\description{
  \code{RandomOverSampler} is an \link[R6:R6Class]{R6} class implementing
  random over-sampling with optional smoothed bootstrap (ROSE method),
  ported from the Python \code{imbalanced-learn} library.
}
\format{
  An \code{\link[R6:R6Class]{R6}} class. Create an instance with
  \code{RandomOverSampler$new(sampling_strategy = "auto", shrinkage = NULL)}:
  \itemize{
    \item \code{sampling_strategy}: A string defining which classes should be
    over-sampled. \code{"auto"} over-samples every non-majority class up to the
    size of the majority class (default); \code{"minority"} over-samples only the
    smallest class.
    \item \code{shrinkage}: Optional non-negative numeric value controlling the
    shrinkage applied for smoothed bootstrap (ROSE method). If \code{NULL} (default),
    standard random sampling with replacement is performed without perturbation.
  }
  
  After creation, call the \code{fit_resample(X, y)} method:
  \code{X} is a feature matrix (one row per sample, one column per feature) and
  \code{y} is a \code{\link{factor}} of class labels.
  \code{length(y)} should be equal to \code{nrow(X)}. 
  It returns a \code{list} with two elements: \code{X}, the feature matrix including
  the over-sampled instances, and the corresponding \code{y}.

  \code{RandomOverSampler} inherits from the abstract classes
  \code{BaseOverSampler} and \code{BaseSampler}.
}
\author{
Aman Kashyap
}
\examples{
set.seed(1)
## 10 positive vs 40 negative
y <- factor(c(rep("positive", 10), rep("negative", 40)))
X <- matrix(rnorm(100), ncol = 2) # 50 samples, 2 features
sampler <- mlr3torchAUM::RandomOverSampler$new(sampling_strategy = "auto")
res <- sampler$fit_resample(X, y)
nrow(res$X)   # 50 original + 30 duplicated = 80
table(res$y)  # balanced: positive = 40, negative = 40
}
