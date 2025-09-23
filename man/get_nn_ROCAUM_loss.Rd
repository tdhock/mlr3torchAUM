\name{get_nn_ROCAUM_loss}
\alias{get_nn_ROCAUM_loss}
\title{
  torch module class for ROC AUM loss
}
\description{
  Instantiating this loss function yields a torch loss function which
  can be used as a surrogate for ROC curve optimization.
}
\usage{
get_nn_ROCAUM_loss(e)
}
\arguments{
  \item{e}{\code{NULL} (default) or environment in which to store counts
    of loss function evaluations, zeros, and batches with all one class.}
}
\value{
  torch module instance for ROC AUM loss
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
  count_env <- new.env()
  loss_class <- mlr3torchAUM::get_nn_ROCAUM_loss(count_env)
  loss_fun <- loss_class()
  (loss_tensor <- loss_fun(pred_vec, label_tensor))
  with(count_env, data.frame(evals, zeros, all_one_class))
  loss_tensor$backward()
  sapply(linear_model$parameters, "[[", "grad")
}

}
