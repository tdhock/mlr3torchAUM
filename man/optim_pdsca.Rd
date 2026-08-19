\name{optim_pdsca}
\alias{optim_pdsca}
\title{PDSCA Optimizer for Compositional AUC Training}
\description{
A \code{torch} optimizer porting LibAUC's PDSCA (Primal-Dual Stochastic
Compositional Adaptive) algorithm for the network weights, used with
\code{\link{nn_CompositionalAUC_loss}} and \code{\link{make_pdsca_callback}}.
Apply gradient clamping, weight decay and an epoch-wise proximal
term to each step in \code{\link{nn_CompositionalAUC_loss}. 
The optimizer also exposes \code{update_regularizer()} (set the proximal
reference to the epoch average) and \code{update_lr()} (decay both learning
rates), called once per epoch by \code{\link{make_pdsca_callback}}.
Integrated into \code{mlr3torch} via
\code{mlr3torch::as_torch_optimizer(optim_pdsca)}.
}
\usage{
optim_pdsca(params, lr0, lr, clamp_value, weight_decay, epoch_decay,
            weight_momentum, momentum, decay_factor0, decay_factor)
}
\arguments{
\item{params}{list of tensors to optimize (the network parameters)}
\item{lr0}{learning rate of the \strong{inner}, cross-entropy step.}
\item{lr}{learning rate of the \strong{outer}, AUC-margin step.}
\item{clamp_value}{gradient clamping threshold; gradients are clamped to
\code{[-clamp_value, clamp_value]}.}
\item{weight_decay}{L2 regularization strength}
\item{epoch_decay}{strength of the proximal term}
\item{weight_momentum}{EMA weight for averaging the \strong{parameters} on the
cross-entropy pass (\code{beta1} upstream).}
\item{momentum}{EMA weight for averaging the \strong{gradients} on the
AUC-margin pass (\code{beta2} upstream).}
\item{decay_factor0}{\code{lr0} is divided by this after each epoch}
\item{decay_factor}{\code{lr} is divided by this after each epoch}
}
\value{A \code{torch} optimizer with functions \code{step()},
\code{update_lr()} and \code{update_regularizer()}.}
\note{
\strong{This optimizer is not usable on its own.} It has to be paired with
\code{\link{make_pdsca_callback}}, which injects the loss module so that
\code{step()} can tell which pass the current batch belongs to, and with
\code{\link{nn_CompositionalAUC_loss}}, which carries the counter driving the
alternation. Using any other loss raises an error at the start of training.

\strong{Both momentum arguments follow LibAUC's convention, which is the
reverse of PyTorch's.} The update is
\code{buf <- (1 - beta) * buf + beta * new}, so \code{beta} is the weight on
the \emph{new} value: larger means \emph{less} smoothing.
Setting either weight to \code{0} disables that averaging entirely and falls
back to plain SGD, matching LibAUC upstream; \code{1} is equivalent numerically.
}
\references{
Yuan, Z., Guo, Z., Chawla, N., Yang, T. (2022).
Compositional Training for End-to-End Deep AUC Maximization. ICLR 2022.
\url{https://openreview.net/forum?id=gPvB4pdu_Z}
}
\author{Wei Cao}
\seealso{\code{\link{nn_CompositionalAUC_loss}},
\code{\link{make_pdsca_callback}}, \code{\link{optim_pesg}}}
