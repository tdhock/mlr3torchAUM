minority_support_vectors <- function(svm_index, y, target_class) {
  return(svm_index[as.character(y[svm_index]) == target_class])
}

fit_svm <- function(X, y) {
  if (!requireNamespace("e1071")) stop("e1071 not installed ")
  gamma <- 1 / (ncol(X) * (mean(X^2) - mean(X)^2))
  e1071::svm(x = X, y = y, kernel = "radial", cost = 1, gamma = gamma, scale = FALSE)
}
