\name{nn_squared_hinge_loss}
\alias{nn_squared_hinge_loss}
\alias{SquaredHingeLoss}
\alias{torch_loss_sq_hinge_loglinear}
\alias{sq_hinge_loglinear}
\title{
  All-Pairs Squared Hinge Log-Linear Loss
}
\description{
  Computes the all-pairs squared hinge loss over all positive-negative
  pairs as a convex surrogate for AUC, in \eqn{O(n \log n)} time instead
  of the naive \eqn{O(n^2)} double loop.
}
\usage{
nn_squared_hinge_loss(margin = 1)
SquaredHingeLoss(pred_tensor, label_tensor, margin = 1)
}
\arguments{
  \item{pred_tensor}{tensor of real-valued predicted scores, larger for
    more likely to be positive class.}
  \item{label_tensor}{1 for positive class, -1 or other values for negative class.}
  \item{margin}{non-negative scalar margin hyper-parameter (default 1).}
}
\details{
  This loss function is also registered in the \code{mlr3torch} package
  under the dictionary key \code{"sq_hinge_loglinear"}. Once this package is loaded, 
  the loss wrapper can be retrieved using \code{mlr3torch::t_loss("sq_hinge_loglinear")}.
}
\value{
  \code{SquaredHingeLoss()} returns a scalar torch tensor (total
  loss, not averaged), suitable for use with automatic differentiation.
  \code{nn_squared_hinge_loss()} returns a torch loss module with
  buffers \code{evals}, \code{zeros}, and \code{all_one_class},
  suitable for use as the \code{loss} parameter of a
  \code{TorchLearner}.
  Call \code{$state_dict()} on its instances to get
  \describe{
    \item{evals}{times the loss has been called.}
    \item{zeros}{times the loss returned a value equal to zero.}
    \item{all_one_class}{times the loss has been called with a batch of
      labels all from one class.}
  }
}
\references{
  Rust and Hocking. A Log-linear Gradient Descent Algorithm for
  Unbalanced Binary Classification using the All Pairs Squared Hinge
  Loss. arXiv:2302.11062, 2023.
}
\examples{

if(torch::torch_is_installed()){
  n_obs <- 4
  n_feat <- 5
  torch::torch_manual_seed(1)
  features <- torch::torch_rand(n_obs, n_feat)
  label_tensor <- torch::torch_tensor(c(1, -1, 1, -1))
  linear_model <- torch::nn_linear(n_feat, 1)
  pred_vec <- linear_model(features)
  loss_fun <- mlr3torchAUM::nn_squared_hinge_loss(margin = 1)
  (loss_tensor <- loss_fun(pred_vec, label_tensor))
  loss_fun$state_dict()
  loss_tensor$backward()
  sapply(linear_model$parameters, "[[", "grad")
}

}
