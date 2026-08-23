\name{ADASYN}
\alias{ADASYN}
\docType{data}
\title{
  ADASYN Over-Sampler
}
\description{
  \code{ADASYN} is an \link[R6:R6Class]{R6} class implementing the
  Adaptive Synthetic (ADASYN) sampling approach for imbalanced datasets,
  ported from the Python \code{imbalanced-learn} library. Unlike standard SMOTE,
  ADASYN adaptively generates more synthetic samples for minority points that
  are harder to learn (i.e. surrounded by majority-class neighbors).
}
\format{
  An \code{\link[R6:R6Class]{R6}} class. Create an instance with
  \code{ADASYN$new(sampling_strategy = "auto", k_neighbors = 5)}:
  \itemize{
    \item \code{sampling_strategy}: A string field defining samples of 
    which classes should be synthesized. \code{"auto"}
    over-samples every non-majority class up to 
    the size of the majority class. Default; \code{"minority"}
    over-samples only the smallest class.
    Only one class will be selected.
    \item \code{k_neighbors}: The hyperparameter \code{k} for nearest neighbor density estimation and sample generation.
  }
  
  After creation, call the \code{fit_resample(X, y)} method:
  \code{X} is a matrix of features (one row per sample, 
  one column per feature) and
  \code{y} is a \code{\link{factor}} of class labels.
  \code{length(y)} should be equal to \code{nrow(X)}. 
  It returns a \code{list}
  with two elements: \code{X}, the feature matrix including the original
  and the synthesized ones, and the corresponding \code{y}.

  \code{ADASYN} inherits from the abstract classes
  \code{BaseADASYN}, \code{BaseOverSampler} and \code{BaseSampler}.
}

\examples{
set.seed(1)
## 10 positive vs 40 negative
y <- factor(c(rep("positive", 10), rep("negative", 40)))
X <- matrix(rnorm(100), ncol = 2) # 50 samples, each two features
sampler <- mlr3torchAUM::ADASYN$new(sampling_strategy = "auto", k_neighbors = 5)
res <- sampler$fit_resample(X, y)
nrow(res$X)   # original + synthetic samples
table(res$y)  # balanced output
}
