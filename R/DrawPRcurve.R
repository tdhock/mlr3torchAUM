#' Plot Precision-Recall Curve
#'
#' Computes and visualizes the precision-recall (PR) curve.
#'
#' @param pred_tensor 1D torch tensor of predictions (real-valued scores)
#' @param label_tensor 1D torch tensor of labels (-1 = negative, 1 = positive)
#'
#' @return A ggplot2 object displaying the PR curve.
#'
#' @seealso [PR_curve()]
#'
#' @importFrom torch as_array
#' @importFrom ggplot2 ggplot aes geom_line theme_minimal labs coord_cartesian
#' @export

Draw_PR_curve <- function(pred_tensor, label_tensor) {
  pr <- PR_curve(pred_tensor, label_tensor)
  precision <- torch::as_array(pr$precision)
  recall <- torch::as_array(pr$recall)
  df <- data.frame(recall = recall, precision = precision)
  ggplot2::ggplot(df, ggplot2::aes(x = recall, y = precision)) +
    ggplot2::geom_line(color = "blue", size = 1) +
    ggplot2::theme_minimal() +
    ggplot2::labs(title = "Precision-Recall Curve", x = "Recall", y = "Precision") +
    ggplot2::coord_cartesian(xlim = c(0, 1), ylim = c(0, 1))
}
