\name{make_pdsca_callback}
\alias{make_pdsca_callback}
\title{PDSCA callback for CompositionalAUCLoss}
\description{
Creates an \code{mlr3torch} callback used together with
\code{optim_pdsca} and \code{nn_CompositionalAUC_loss}.
}
\usage{
make_pdsca_callback(lr, clamp_value, weight_decay, epoch_decay,
                    momentum, decay_factor)
}
\arguments{
\item{lr}{learning rate for the loss parameters \code{a}, \code{b} and
\code{alpha}.}
\item{clamp_value}{gradient clamping threshold.}
\item{weight_decay}{L2 regularization strength.}
\item{epoch_decay}{strength of the proximal term}
\item{momentum}{EMA weight for the descent buffer of \code{a} and \code{b};
same convention as in \code{\link{optim_pdsca}}.}
\item{decay_factor}{the learning rate is divided by this after each epoch}
}
\value{An \code{mlr3torch} \code{TorchCallback}.}
\note{
These arguments govern \strong{only} the loss parameters \code{a}, \code{b} and
\code{alpha}. The network weights are governed by the separate arguments of
\code{\link{optim_pdsca}}. So the complete usage of PDSCA optimizer
should be together with \code{\link{optim_pdsca}}. 
}
\author{Wei Cao}
\seealso{\code{\link{nn_CompositionalAUC_loss}}, \code{\link{optim_pdsca}},
\code{\link{make_pesg_callback_full}}}
\examples{
if(torch::torch_is_installed() && requireNamespace("mlr3torch", quietly = TRUE)){
cb <- mlr3torchAUM::make_pdsca_callback(
  lr = 0.1, clamp_value = 1, weight_decay = 1e-4,
  epoch_decay = 1e-3, momentum = 0.9, decay_factor = 2)
}
}
