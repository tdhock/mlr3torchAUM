\name{nn_CompositionalAUC_loss}
\alias{nn_CompositionalAUC_loss}
\title{Compositional AUC Loss}
\description{
R/torch port of \code{CompositionalAUCLoss} from LibAUC. It alternates
between a cross-entropy pass and an \code{\link{AUCM}} pass on a fixed
schedule, which realizes the compositional objective
\code{L_AUC(w - lr0 * grad L_CE(w))}: the inner cross-entropy step moves the
weights, the outer AUC-margin step is evaluated at the moved position.
Used with \code{\link{optim_pdsca}} plus \code{\link{make_pdsca_callback}}.
}
\usage{
nn_CompositionalAUC_loss(margin = 1, k = 1, add_sigmoid = TRUE)
}
\arguments{
\item{margin}{desired gap between positive and negative score means, used by
the AUCM pass.}
\item{k}{number of consecutive batches spent on each pass. \code{k = 1} gives
CE, AUCM, CE, AUCM, ...; \code{k = 2} gives CE, CE, AUCM, AUCM, ...}
\item{add_sigmoid}{if \code{TRUE} (default), \code{torch_sigmoid()} is applied
to \code{pred} before anything else, so the module expects \strong{logits} which
a normal \code{mlr3torch} network head emits. Set it to \code{FALSE} only
when the network already ends in a sigmoid and therefore emits probabilities.}
}
\value{
A torch module usable as the \code{loss} of a \code{TorchLearner}.
}
\note{
\strong{Feed logits, not probabilities.} Both passes need scores in
\code{[0, 1]}. However, passing probabilities while
\code{add_sigmoid = TRUE} applies the sigmoid
twice, which is not correct. Attention needed.
}
\references{
Yuan, Z., Guo, Z., Chawla, N., Yang, T. (2022).
Compositional Training for End-to-End Deep AUC Maximization. ICLR 2022.
\url{https://openreview.net/forum?id=gPvB4pdu_Z}
}
\author{Wei Cao}
\seealso{\code{\link{optim_pdsca}}, \code{\link{make_pdsca_callback}},
\code{\link{nn_AUCM_loss}}}
\examples{
if(torch::torch_is_installed()){
label <- torch::torch_tensor(c(0, 0, 1, 1))
logit <- torch::torch_tensor(qlogis(c(0.1, 0.4, 0.35, 0.8)))
loss_fun <- mlr3torchAUM::nn_CompositionalAUC_loss(margin = 1, k = 1)
loss_fun(logit, label)   # first call: the cross-entropy pass
loss_fun(logit, label)   # second call: the AUC-margin pass
}
}
