\name{make_pesg_callback}
\alias{make_pesg_callback}
\title{PESG Callback for Training the AUCM Loss}
\description{
Creates an \code{mlr3torch} callback that updates the trainable parameters
\code{a}, \code{b}, \code{alpha} of \code{\link{nn_AUCM_loss}} during
training (the mlr3torch optimizer only updates network weights). At each
batch (\code{on_after_backward}) it descends \code{a}, \code{b} (SGD or
Adam decoupled with network params) and ascends \code{alpha}.
}
\usage{
make_pesg_callback(lr = 0.05, mode = "sgd", ...)
}
\arguments{
\item{lr}{learning rate for the loss parameters \code{a}, \code{b}, \code{alpha}.}
\item{mode}{\code{"sgd"} (plain SGD on \code{a}, \code{b}) or \code{"adam"} (Adam).}
\item{...}{extra Adam hyperparameters (\code{betas}, \code{eps}) forwarded
    to the Adam update when \code{mode = "adam"}.}
}
\value{
An \code{mlr3torch} \code{TorchCallback}, for the \code{callbacks} field
of a \code{TorchLearner}.
}
\author{Wei Cao}
\examples{
if(torch::torch_is_installed() && requireNamespace("mlr3torch")){
cb_sgd <- mlr3torchAUM::make_pesg_callback(lr = 0.05, mode = "sgd")
cb_adam <- mlr3torchAUM::make_pesg_callback(lr = 0.001, mode = "adam", betas = c(0.9, 0.999), eps = 1e-8)
}
}