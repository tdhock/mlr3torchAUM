count_class <- function(y) {
  table_y <- table(y)
  table_y <- table_y[table_y > 0]
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

knn_index <- function(query, data, k) {
  if (nrow(data) < k + 1) stop(sprintf("need at least k+1=%d samples, got %d", k + 1, nrow(data)))
  # assume query always included in data
  return(RANN::nn2(data, query, k = k + 1)$nn.idx[, -1, drop = FALSE])
}

generate_samples <- function(X, nn_num, rows, cols, steps, nn_data = X) {
  num_neigh <- length(rows)
  if (num_neigh != length(cols) || num_neigh != length(steps)) stop("rows, cols and steps not consistent")
  if (any(rows > nrow(X))) stop("illegal row number")
  if (any(cols > ncol(nn_num))) stop("illegal col number")
  x <- X[rows, , drop = FALSE]
  neighs <- nn_data[nn_num[cbind(rows, cols)], , drop = FALSE]
  return(x + steps * (neighs - x))
}

make_samples <- function(X, nn, n_to_generate, nn_data = X, step_size = 1) {
  rows <- sample.int(nrow(X), n_to_generate, replace = TRUE)
  cols <- sample.int(ncol(nn), n_to_generate, replace = TRUE)
  steps <- step_size * runif(n_to_generate, 0, 1)
  return(generate_samples(X, nn, rows, cols, steps, nn_data = nn_data))
}

in_danger_noise <- function(nn_idx, y, target_class, kind = "danger") {
  m <- ncol(nn_idx)
  is_maj <- matrix(as.character(y)[nn_idx] != target_class, nrow = nrow(nn_idx))
  n_maj <- rowSums(is_maj)
  if (kind == "danger") {
    return(n_maj >= m / 2 & n_maj < m)
  }
  return(n_maj == m)
}

BaseSMOTE <- R6::R6Class(
  "BaseSMOTE",
  inherit = BaseOverSampler,
  public = list(
    k_neighbors = NULL,
    nn_k_ = NULL,
    initialize = function(sampling_strategy = "auto", k_neighbors = 5) {
      super$initialize(sampling_strategy)
      self$k_neighbors <- k_neighbors
    }
  ),
  private = list(
    .validate_estimator = function() {
      if (!is.numeric(self$k_neighbors) || length(self$k_neighbors) != 1 ||
        self$k_neighbors < 1 || self$k_neighbors != round(self$k_neighbors)) {
        stop("k_neighbors must be a positive integer")
      }
      self$nn_k_ <- function(query, data) knn_index(query, data, self$k_neighbors)
    },
    .make_samples = function(X, nn, n_to_generate, nn_data = X, step_size = 1) make_samples(X, nn, n_to_generate, nn_data, step_size)
  )
)

SMOTE <- R6::R6Class(
  "SMOTE",
  inherit = BaseSMOTE,
  private = list(
    .fit_resample = function(X, y) {
      private$.validate_estimator()
      results <- lapply(names(self$sampling_strategy_), function(class_name) {
        n_to_generate <- self$sampling_strategy_[[class_name]]
        X_within_class <- X[as.character(y) == class_name, , drop = FALSE]
        generated <- private$.make_samples(X_within_class, self$nn_k_(X_within_class, X_within_class), n_to_generate)
        return(list(X = generated, y = rep(class_name, n_to_generate)))
      })
      return(list(
        X = do.call(rbind, c(list(X), lapply(results, function(result) result$X))),
        y = factor(c(as.character(y), unlist(lapply(results, function(result) result$y))),
          levels = levels(y)
        )
      ))
    }
  )
)
