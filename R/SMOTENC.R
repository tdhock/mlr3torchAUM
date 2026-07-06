split_features <- function(X, categorical_features = "auto") {
  num_features <- ncol(X)
  if (identical(categorical_features, "auto")) {
    categorical_features <- which(vapply(X, function(features) {
      is.character(features) || is.factor(features)
    }, logical(1)))
  } else if ((is.character(categorical_features) &&
    anyNA(categorical_features <- match(categorical_features, colnames(X)))
  ) || (!is.character(categorical_features) &&
    any(categorical_features < 1 | categorical_features > num_features))) {
    stop("illegal category name or index")
  }
  continuous_features <- setdiff(seq_len(num_features), categorical_features)
  if (length(categorical_features) == 0) stop("No categorical features, use SMOTE instead")
  if (length(continuous_features) == 0) stop("No continuous features, use SMOTEN instead")
  return(list(
    categorical = as.matrix(X[, categorical_features, drop = FALSE]),
    continuous = as.matrix(X[, continuous_features, drop = FALSE])
  ))
}

median_std <- function(X_cont) {
  if (nrow(X_cont) == 0 || ncol(X_cont) == 0) {
    return(0)
  }
  median(apply(X_cont, 2, function(column) {
    sqrt(sum((column - mean(column))^2) / length(column))
  }))
}


smotenc_distances <- function(continuous, categorical) {
  # sqrt of (continuous Euclidean^2 + median_std^2 * category mismatches)
  med_std <- median_std(continuous)
  cont_dist_matrix <- as.matrix(dist(continuous))
  category_mismatches <- Reduce("+", lapply(seq_len(ncol(categorical)), function(col_idx) {
    outer(categorical[, col_idx], categorical[, col_idx], "!=")
  }))
  sqrt(cont_dist_matrix^2 + med_std^2 * category_mismatches)
}
