\name{nn_IMCP_loss}
\alias{nn_IMCP_loss}
\alias{nn_MCP_loss}
\title{
  (Imbalanced) Multiclass Classification Performance (MCP/IMCP) losses
}
\description{
  Differentiable \code{torch} losses that maximize the area under the MCP
  curve, or under its variant for imbalanced tasks IMCP curve. Both return
  \code{1 - area}, so minimizing the loss maximizes the area. The IMCP
  variant weights each sample by the inverse frequency of its class, so the
  loss is not dominated by the majority classes. 
}
\usage{
nn_IMCP_loss()
nn_MCP_loss()
}
\arguments{
  \item{pred_tensor}{Argument of the modules' \code{forward()} method: a
    \code{torch} tensor of raw scores (logits) of shape
    \code{(n_samples, n_classes)}.}
  \item{label_tensor}{Argument of the modules' \code{forward()} method: a
    \code{torch} long tensor of 1-based true class labels (values \code{1}
    to \code{n_classes}), of length \code{n_samples}.}
}
\value{
  A \code{torch} loss module generator. Suitable for use as the
  \code{loss} parameter of a \code{TorchLearner}.
}
\references{
  Aguilar-Ruiz, J. S. and Michalak, M. (2024) Classification performance
  assessment for imbalanced multiclass data. Scientific Reports 14:10759.
  \doi{10.1038/s41598-024-61365-z}
}
\author{
Wei Cao
}
\seealso{
  \code{\link{mcp_curve}} for the curves and areas themselves,
  \code{\link{MeasureClassifIMCP}} for the matching \code{mlr3} measures.
}
\examples{

if (torch::torch_is_installed()) {
  ## 4 samples, 3 classes. Raw logits, NOT probabilities.
  logits <- torch::torch_tensor(rbind(
    c(2.0, 0.5, 0.1),
    c(0.3, 1.4, 0.3),
    c(0.2, 1.6, 0.2),
    c(0.1, 0.3, 2.6)), requires_grad = TRUE)
  label <- torch::torch_tensor(c(1L, 1L, 2L, 3L), dtype = torch::torch_long())

  loss_fun <- nn_IMCP_loss()
  (loss_tensor <- loss_fun(logits, label))
  loss_tensor$backward()
  logits$grad

  torch::as_array(nn_MCP_loss()(logits, label))
}

}
