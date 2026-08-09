\name{nn_AUCM_loss}
\alias{nn_AUCM_loss}
\alias{AUCM}
\title{AUCM Loss}
\description{
Faithful R/torch port of the AUC-Margin loss from LibAUC: a surrogate for AUC
maximization.
Train with \code{\link{make_pesg_callback}}.
}
\usage{
nn_AUCM_loss(margin = 1, version = "v1", imratio = NULL)
AUCM(pred_tensor, label_tensor, a = 0, b = 0, alpha = 0, margin = 1,
     version = "v1", imratio = NULL)
}
\arguments{
\item{margin}{desired gap between positive and negative score means. Its
scale has to match the scale of \code{pred_tensor}}
\item{version}{\code{"v1"} (default) weights each term by the class prior
\code{p}; \code{"v2"} drops \code{p} and averages within each class.}
\item{imratio}{class prior \code{p} for \code{"v1"}, as a single number in
\code{(0, 1)}. \code{NULL} (default) estimates it from each batch. Ignored
by \code{"v2"}, which has no class prior.}
\item{pred_tensor}{tensor of predicted scores, with the last layer 
as \code{sigmoid} to fix the values within (0,1)}
\item{label_tensor}{tensor of labels, 1 for positive, 0 for negative}
\item{a}{center of positive scores.}
\item{b}{center of negative scores.}
\item{alpha}{dual variable; maximized, constrained \code{>= 0, <=999}.}
}
\details{
\code{"v1"} averages every term over the whole batch and weights the terms by
the class prior \code{p}, so the loss scale shrinks with the imbalance ratio.
\code{"v2"} instead averages within each class, which removes \code{p} as an
overall factor: the two versions differ exactly by that factor,
\code{v2 == v1 / (p * (1 - p))}. A learning rate therefore transfers across
data sets of different imbalance under \code{"v2"}. In exchange, \code{"v2"}
requires every batch to contain both classes; a single-class batch makes a
class mean \code{0/0} and the loss \code{NaN}, which then propagates into the
weights. Use a batch scheme that guarantees both classes when training with
\code{"v2"}.

Our \code{"v2"} follows the paper. It differs numerically from LibAUC 2.0.1's
\code{version='v2'}, which has two implementation flaws that we do not
reproduce. First, LibAUC subtracts the positive mean inside its masked mean of
the cross term, so the positive positions stop being zero and enter the
denominator; its cross term becomes
\code{margin + (1 - p) * E[s|y=0] - E[s|y=1]} instead of
\code{margin + E[s|y=0] - E[s|y=1]}, a loss offset of \code{-2 * alpha * p *
E[s|y=0]}. Second, LibAUC divides by \code{count_nonzero(x)} rather than the
class size, so a sample whose term evaluates to exactly zero -- a score
sitting on its own class center, which happens once sigmoid saturates -- is
dropped from the denominator, and a whole class landing on its center yields
\code{NaN} where the correct answer is a zero within-class variance.
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