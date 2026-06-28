\name{make_pesg_callback_full}
\alias{make_pesg_callback_full}
\title{Full PESG Callback for the AUCM Loss}
\description{
Creates an \code{mlr3torch} callback that trains the \code{\link{nn_AUCM_loss}}
parameters with the full PESG rule, matching \code{\link{optim_pesg}}. At each
batch it updates \code{a}, \code{b} via the PESG primal step (clamp + weight
decay + proximal + EMA momentum) and ascends the dual variable \code{alpha};
at each epoch end it refreshes the proximal reference and decays the learning
rate, remaining consistent with the network optimizer. Must be paired with
\code{\link{optim_pesg}} as the learner's optimizer for network weights.
}
\usage{
make_pesg_callback_full(lr, clamp_value, weight_decay, epoch_decay,
                        momentum, decay_factor)
}
\arguments{
\item{lr}{learning rate for the loss parameters \code{a}, \code{b} and
\code{alpha}.}
\item{clamp_value}{gradient clamping threshold.}
\item{weight_decay}{L2 regularization strength.}
\item{epoch_decay}{strength of the proximal term}
\item{momentum}{EMA momentum weight.}
\item{decay_factor}{the learning rate is divided by this after each epoch}
}
\value{An \code{mlr3torch} \code{TorchCallback}.}
\author{Wei Cao}
\seealso{\code{\link{nn_AUCM_loss}}, \code{\link{optim_pesg}}}