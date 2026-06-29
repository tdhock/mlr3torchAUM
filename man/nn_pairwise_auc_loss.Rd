\name{nn_pairwise_auc_loss}
\alias{nn_pairwise_auc_loss}
\alias{pairwise_auc_surrogates}
\alias{pairwise_auc_surrogate}
\title{
  Pairwise AUC Surrogate Loss
}
\description{
  Computes a pairwise AUC surrogate loss over positive-negative pairs using one of several surrogate loss functions. The implementation is
  adapted from the LibAUC library.
}
\usage{
nn_pairwise_auc_loss(surr_loss = "logistic", margin = 1.0, scale = 1.0)
pairwise_auc_surrogates
}
\arguments{
  \item{surr_loss}{character string specifying the surrogate loss function.
    Must be one of "squared", "squared_hinge", "hinge", "logistic", or
    "barrier_hinge".}
  \item{margin}{non-negative scalar margin hyper-parameter used by the
    "squared", "squared_hinge", "hinge", and "barrier_hinge" surrogate losses. (default 1.0).}
  \item{scale}{non-negative scalar scale hyper-parameter used by the
    "logistic" and "barrier_hinge" surrogate losses. (default 1.0).}
}
\details{
  This loss function is also registered in the \code{mlr3torch} package
  under the dictionary key \code{"pairwise_auc_surrogate"}. Once this package is loaded,
  the loss wrapper can be retrieved using \code{mlr3torch::t_loss("pairwise_auc_surrogate")}.
}
\value{
  \code{nn_pairwise_auc_loss()} returns a torch loss module with
  buffers \code{evals}, \code{zeros}, and \code{all_one_class},
  suitable for use as the \code{loss} parameter of a
  \code{TorchLearner}.
  \code{pairwise_auc_surrogates} is a named list of the mathematical
  surrogate loss functions.
  Call \code{$state_dict()} on loss module instances to get:
  \describe{
    \item{evals}{times the loss has been called.}
    \item{zeros}{times the loss returned a value equal to zero.}
    \item{all_one_class}{times the loss has been called with a batch of
      labels all from one class.}
  }
}
\references{
  Yuan et al. LibAUC: A Deep Learning Library for X-Risk Optimization.
  arXiv:2306.03065, 2023.

  Charoenphakdee et al. On Symmetric Losses for Learning from Corrupted
  Labels. Proceedings of the 36th International Conference on Machine
  Learning, PMLR 97:961-970, 2019. (Barrier Hinge Loss)
}
\examples{

if(torch::torch_is_installed()){
  n_obs <- 4
  n_feat <- 5
  torch::torch_manual_seed(1)
  features <- torch::torch_rand(n_obs, n_feat)
  label_tensor <- torch::torch_tensor(c(1, 0, 1, 0))
  linear_model <- torch::nn_linear(n_feat, 1)
  pred_vec <- linear_model(features)
  loss_fun <- mlr3torchAUM::nn_pairwise_auc_loss(surr_loss = "logistic", scale = 1)
  (loss_tensor <- loss_fun(pred_vec, label_tensor))
  loss_fun$state_dict()
  loss_tensor$backward()
  sapply(linear_model$parameters, "[[", "grad")
}

}
