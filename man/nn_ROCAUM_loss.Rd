\name{nn_ROCAUM_loss}
\alias{nn_ROCAUM_loss}
\alias{ROCAUM}
\title{
  Area Under Minimum of False Positive and False Negative Rates
}
\description{
  Instantiating this loss function yields a torch loss function which
  can be used as a surrogate for ROC curve optimization.
}
\usage{
nn_ROCAUM_loss()
ROCAUM(pred_tensor, label_tensor)
}
\arguments{
  \item{pred_tensor}{tensor of real-valued predicted scores, larger for
    more likely to be positive class.}
  \item{label_tensor}{real-valued tensor of labels, 1 for positive
    class, negative otherwise.}
}
\value{
  \code{ROCAUM()} is a torch tensor (scalar loss value), suitable for
  use as loss function for ROC curve optimization.
  \code{nn_ROCAUM_loss} is suitable for use as the \code{loss} parameter
  of a \code{TorchLearner}. Call \code{$state_dict()} on its instances
  to get
  \describe{
    \item{evals}{times the loss has been called.}
    \item{zeros}{times the loss returned a value equal to zero.}
    \item{all_one_class}{times the loss has been called with a batch of
      labels all from one class.}
  }
}
\references{
  Hillman and Hocking. Optimizing ROC Curves with a Sort-Based Surrogate
  Loss for Binary Classification and Changepoint Detection. Journal of
  Machine Learning Research 24(70), 2023.
}
\author{
  Toby Dylan Hocking
}
\examples{

if(torch::torch_is_installed()){
  n_obs <- 2
  n_feat <- 5
  torch::torch_manual_seed(1)
  features <- torch::torch_rand(n_obs, n_feat)
  label_tensor <- torch::torch_tensor(c(1,0))
  linear_model <- torch::nn_linear(n_feat, 1)
  pred_vec <- linear_model(features)
  loss_fun <- mlr3torchAUM::nn_ROCAUM_loss()
  (loss_tensor <- loss_fun(pred_vec, label_tensor))
  loss_fun$state_dict()
  loss_tensor$backward()
  sapply(linear_model$parameters, "[[", "grad")
}

}
