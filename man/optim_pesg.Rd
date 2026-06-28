\name{optim_pesg}
\alias{optim_pesg}
\title{PESG Optimizer for AUCM Training}
\description{
A \code{torch} optimizer porting LibAUC's PESG (Proximal Epoch-wise Stochastic
Gradient) for the network weights training with \code{\link{nn_AUCM_loss}}.
Each step applies gradient clamping, weight decay, an epoch-wise proximal term
and EMA momentum. It also exposes \code{update_regularizer()} (update the
proximal reference to the epoch average) and \code{update_lr()} (decay the
learning rate), called once per epoch by \code{\link{make_pesg_callback_full}}
inside the hook \code{on_epoch_end}.
Integrated into \code{mlr3torch} via \code{mlr3torch::as_torch_optimizer(optim_pesg)}.
}
\usage{
optim_pesg(params, lr, clamp_value, weight_decay, epoch_decay, momentum, decay_factor)
}
\arguments{
\item{params}{list of tensors to optimize (the network parameters)}
\item{lr}{learning rate.}
\item{clamp_value}{gradient clamping threshold; gradients are clamped to
\code{[-clamp_value, clamp_value]}.}
\item{weight_decay}{L2 regularization strength}
\item{epoch_decay}{strength of the proximal term}
\item{momentum}{EMA momentum weight for the descent buffer.}
\item{decay_factor}{the learning rate is divided by this after each epoch}
}
\value{A \code{torch} optimizer with functions \code{step()},
\code{update_lr()} and \code{update_regularizer()}.}
\references{
Yuan, Z., Yan, Y., Sonka, M., Yang, T. (2021).
Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical
Studies on Medical Image Classification. ICCV 2021. arXiv:2012.03173.
}
\author{Wei Cao}
\seealso{\code{\link{nn_AUCM_loss}}, \code{\link{make_pesg_callback_full}}}