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

generate_samples <- function(X, nn, rows, cols, steps) {
  num_neigh <- length(rows)
  if (num_neigh != length(cols) || num_neigh != length(steps)) stop("rows, cols and steps not consistent")
  if (any(rows > nrow(X))) stop("illegal row number")
  if (any(cols > ncol(nn))) stop("illegal col number")
  if (any(steps > 1 | steps < 0)) stop("illegal step")
  x <- X[rows, , drop = FALSE]
  neighs <- X[nn[cbind(rows, cols)], , drop = FALSE]
  return(x + steps * (neighs - x))
}

make_samples <- function(X, nn, n_to_generate) {
  rows <- sample.int(nrow(X), n_to_generate, replace = TRUE)
  cols <- sample.int(ncol(nn), n_to_generate, replace = TRUE)
  steps <- runif(n_to_generate, 0, 1)
  return(generate_samples(X, nn, rows, cols, steps))
}

fit_resample <- function(X, y, strategy = "auto", k = 5) {
  n_samples <- nrow(X)
  if (n_samples != length(y)) stop("data and label not consistent")
  sampling_strategy <- check_sampling_strategy(y, strategy)
  results <- lapply(names(sampling_strategy), function(class_name) {
    n_to_generate <- sampling_strategy[[class_name]]
    if (n_to_generate != 0) {
      X_within_class <- X[as.character(y) == class_name, , drop = FALSE]
      nn <- knn_within_class(X_within_class, k)
      generated <- make_samples(X_within_class, nn, n_to_generate)
      return(list(X = generated, y = rep(class_name, n_to_generate)))
    }
  })
  return(list(
    X = do.call(rbind, c(list(X), lapply(results, function(result) result$X))),
    y = factor(c(as.character(y), unlist(lapply(results, function(result) result$y))),
      levels = levels(y)
    )
  ))
}
