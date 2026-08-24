\name{RandomUnderSampler}
\alias{RandomUnderSampler}
\docType{data}
\title{
  Random Under-Sampler
}
\description{
  \code{RandomUnderSampler} is an \link[R6:R6Class]{R6} class implementing
  random under-sampling,
  ported from the Python \code{imbalanced-learn} library.
}
\format{
  An \code{\link[R6:R6Class]{R6}} class. Create an instance with
  \code{RandomUnderSampler$new(sampling_strategy = "auto", replacement = FALSE)}:
  \itemize{
    \item \code{sampling_strategy}: A string defining which classes should be
    under-sampled. \code{"auto"} under-samples all classes except the minority
    class down to the size of the minority class (default); \code{"all"} resamples
    all classes.
    \item \code{replacement}: A boolean indicating whether the sample should be 
    drawn with or without replacement. \code{FALSE} by default.
  }
  
  After creation, call the \code{fit_resample(X, y)} method:
  \code{X} is a feature matrix (one row per sample, one column per feature) and
  \code{y} is a \code{\link{factor}} of class labels.
  \code{length(y)} should be equal to \code{nrow(X)}. 
  It returns a \code{list} with two elements: \code{X}, the feature matrix containing
  the under-sampled instances, and the corresponding \code{y}.
}
\author{
Aman Kashyap
}
\examples{
set.seed(1)
## 40 positive vs 10 negative
y <- factor(c(rep("positive", 40), rep("negative", 10)))
X <- matrix(rnorm(100), ncol = 2) # 50 samples, 2 features
sampler <- mlr3torchAUM::RandomUnderSampler$new(sampling_strategy = "auto")
res <- sampler$fit_resample(X, y)
nrow(res$X)   # 10 positive + 10 negative = 20
table(res$y)  # balanced: positive = 10, negative = 10
}
