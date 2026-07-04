\name{SVMSMOTE}
\alias{SVMSMOTE}
\docType{data}
\title{
  SVM-SMOTE Over-Sampler
}
\description{
  \code{SVMSMOTE} is an \link[R6:R6Class]{R6} class implementing the
  support-vector variant of the Synthetic Minority Over-sampling Technique
  (SVM-SMOTE), ported from the Python \code{imbalanced-learn} library.
}
\format{
  An \code{\link[R6:R6Class]{R6}} class. Create an instance with
  \code{SVMSMOTE$new(sampling_strategy = "auto", k_neighbors = 5, m_neighbors = 10, out_step = 0.5)}:
  \itemize{
    \item \code{sampling_strategy} A string field defining which classes to
    over-sample. \code{"auto"} over-samples every non-majority
    class up to the size of the majority class. Default; \code{"minority"}
    over-samples only the smallest class.
    \item \code{k_neighbors} The number of nearest neighbors \emph{within the
    minority class} used to interpolate / extrapolate new samples.
    \item \code{m_neighbors} The number of nearest neighbors \emph{over the
    whole data set} used to classify each minority support vector as
    \emph{noise}, \emph{danger} or \emph{safe}.
    \item \code{out_step} The step size for extrapolating from the
    \emph{safe} support vectors (away from their neighbors).
  }

  After creation, call the \code{fit_resample(X, y)} method:
  \code{X} is a matrix of features (one row per sample, one column per
  feature) and \code{y} is a \code{\link{factor}} of class labels;
  \code{length(y)} must equal \code{nrow(X)}. It returns a \code{list} with
  two elements: \code{X}, the feature matrix including the original and the
  synthesized samples, and the corresponding \code{y}.

  For each over-sampled class, \code{SVMSMOTE} trains an SVM once, keeps that
  class's support vectors, discards the \emph{noise} ones (surrounded by the
  majority), then splits the rest into \emph{danger} and \emph{safe}:
  \emph{danger} support vectors are \emph{interpolated} toward their minority
  neighbors (as in \code{SMOTE}), while \emph{safe} ones are
  \emph{extrapolated} away from them to push the minority frontier outward.
  The split between the two batches is drawn from a \code{Beta(10, 10)}.
  Edge cases: if all of the minority's SVM points are noise, it will not
  generate any new points with a warning message. If not all are noise,
  but either has no safe points or danger points, it will hint the user and
  generate all new samples with the existing genre.
  In fact, the Python \code{imbalanced-learn} does not behave this way: it
  fixes the danger / safe quota in advance, before checking whether either
  batch is empty, so the samples distributed to an empty batch are dropped
  rather than reassigned, in other words, it may generate fewer than the requested number.
  It also leaves the all-noise case unguarded and spits out an error.
  Here for more robust implementation and better user experience, we make
  this improvement.

  \code{SVMSMOTE} inherits from the abstract classes \code{BaseSMOTE},
  \code{BaseOverSampler} and \code{BaseSampler}, identical to the class
  hierarchy of \code{imbalanced-learn}. It requires the \code{e1071} package.
}
\author{
Wei Cao
}
\examples{
if (requireNamespace("e1071", quietly = TRUE)) {
  set.seed(1)
  ## 40 minority points overlapping the boundary of a 60-point majority cloud
  X <- rbind(
    matrix(rnorm(120), ncol = 2),
    matrix(rnorm(80), ncol = 2) + matrix(c(2.2, 0), 40, 2, byrow = TRUE)
  )
  y <- factor(c(rep("maj", 60), rep("min", 40)))
  sampler <- mlr3torchAUM::SVMSMOTE$new(sampling_strategy = "minority", k_neighbors = 3)
  res <- sampler$fit_resample(X, y)
  table(res$y) # minority over-sampled toward the majority size
}
}
