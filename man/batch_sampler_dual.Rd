\name{batch_sampler_dual}
\alias{batch_sampler_dual}
\title{Dual (fixed positive-negative-ratio) batch sampler}
\description{
  R/torch port of LibAUC's \code{DualSampler}: a batch sampler that
  holds the number of positive samples per batch fixed, oversampling the
  minority class once its pool is exhausted. The value returned by this
  function can be used as the \code{batch_sampler} parameter of a
  \code{TorchLearner}, as \code{\link{batch_sampler_stratified}}.
}
\usage{
batch_sampler_dual(batch_size, sampling_rate = 0.5, num_pos = NULL,
                   shuffle = TRUE, random_seed = NULL)
}
\arguments{
  \item{batch_size}{Size of every batch, the use must input it}
  \item{sampling_rate}{Positive ratio in a batch;
    If \code{as.integer(sampling_rate * batch_size)} is \code{0} it will
    be then added to \code{1L}. Exactly one of
    \code{sampling_rate} and \code{num_pos} must be given.
    Passing both is an error.}
  \item{num_pos}{Exact number of positive samples per batch.
    Set \code{sampling_rate = NULL} to use it.}
  \item{shuffle}{if \code{TRUE} (default), shuffle the positive and
    negative index pools at initialization.}
  \item{random_seed}{if not \code{NULL}, sets the global random seed once
    (via \code{set.seed()}) at initialization, matching LibAUC.}
}
\value{
An R6 class created by \code{torch::sampler()}.
}
\references{
\url{https://github.com/Optimization-AI/LibAUC/blob/main/libauc/sampler/sampler.py}
}
\author{Wei Cao}
\examples{
data.table::setDTthreads(1L)
## An imbalanced task: 2\% positive.
n <- 200
labels <- c(rep(1, 4), rep(0, n - 4))
task <- mlr3::TaskClassif$new("imbalanced",
  data.frame(x = seq_len(n),
             y = factor(ifelse(labels == 1, "pos", "neg"),
                        levels = c("neg", "pos"))),
  target = "y", positive = "pos")

## Every batch holds exactly num_pos positives
sampler_class <- mlr3torchAUM::batch_sampler_dual(
  batch_size = 20, sampling_rate = 0.1)
sampler_instance <- sampler_class(list(task = task))
sampler_instance$num_pos
sapply(sampler_instance$batch_list, function(i) sum(i \%in\% which(labels == 1)))

## Typical use case below is as the batch_sampler param in a TorchLearner.
if(torch::torch_is_installed() && requireNamespace("mlr3torch")){
  L <- mlr3torch::LearnerTorchMLP$new(task_type = "classif")
  L$param_set$set_values(
    epochs = 1, batch_size = 20, seed = 1,
    batch_sampler = sampler_class)
  L$train(task)
}
}
