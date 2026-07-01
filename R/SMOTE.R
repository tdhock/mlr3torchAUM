count_class <- function(y) {
  table_y <- table(y)
  return(setNames(as.integer(table_y), names(table_y)))
}

check_sampling_strategy <- function(y, strategy = "auto") {
  count <- count_class(y)
  max_value <- max(count)
  if (strategy == "auto") {
    major_name <- names(which.max(count))[[1]]
    other_count <- count[names(count) != major_name]
    return(max_value - other_count)
  }
  if (strategy == "minority") {
    min_names <- names(which.min(count))
    min_count <- count[names(count) == min_names]
    return(max_value - min_count)
  }
  stop(sprintf("strategy %s not implemented", strategy))
}

knn_within_class <- function(X, k) {
  if (nrow(X) < k + 1) stop(sprintf("need at least k+1=%d samples, got %d", k + 1, nrow(X)))
  return(RANN::nn2(X, X, k = k + 1)$nn.idx[, -1, drop = FALSE])
}
