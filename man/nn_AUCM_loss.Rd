\name{nn_AUCM_loss}
\alias{nn_AUCM_loss}
\alias{AUCM}
\title{AUCM Loss}
\description{
Faithful R/torch port of the AUC-Margin loss from LibAUC: a surrogate for AUC maximization.
Train with \code{\link{make_pesg_callback}}.
}
\usage{
nn_AUCM_loss(margin = 1)
AUCM(pred_tensor, label_tensor, a = 0, b = 0, alpha = 0, margin = 1)
}
\arguments{
\item{margin}{desired gap between positive and negative score means.}
\item{pred_tensor}{tensor of predicted scores after sigmoid, ranging from 0 to 1}
\item{label_tensor}{tensor of labels, 1 for positive, 0 for negative}
\item{a}{center of positive scores.}
\item{b}{center of negative scores.}
\item{alpha}{dual variable; maximized, constrained \code{>= 0, <=999}.}
}
\value{
\code{AUCM()} returns a scalar torch tensor (the loss value).
\code{nn_AUCM_loss} is a torch module for the \code{loss} of a
\code{TorchLearner}; instances hold trainable \code{a}, \code{b},
\code{alpha} (initialized as 0). 
Train with \code{\link{make_pesg_callback}}, not the network optimizer alone.
}
\references{
Yuan, Yan, Sonka, Yang. Large-scale Robust Deep AUC Maximization.
ICCV 2021. arXiv:2012.03173.
}
\author{Wei Cao}
\examples{
if(torch::torch_is_installed()){
pred  <- torch::torch_tensor(c(0.1, 0.4, 0.35, 0.8))
label <- torch::torch_tensor(c(0, 0, 1, 1))
loss_fun <- mlr3torchAUM::nn_AUCM_loss(margin = 1)
(loss_tensor <- loss_fun(pred, label))
loss_tensor$backward()
loss_fun$alpha$grad
}
}