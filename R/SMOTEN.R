get_feature_wise_mode <- function(neighbors) {
  return(apply(neighbors, 2, function(column) {
    counts <- table(column)
    names(counts)[which.max(counts)]
  }))
}

make_samples_nominal <- function(X, nn_indices, n_to_generate) {
  # n_to_generate > 0
  n_samples <- nrow(X)
  if (n_samples == 0) {
    return(matrix(character(0), ncol = ncol(X)))
  }
  if (n_samples != nrow(nn_indices)) stop("sample number not consistent")
  sample_indices <- sample.int(n_samples, n_to_generate, replace = TRUE)
  return(t(sapply(
    sample_indices,
    function(sample_idx) {
      neighbors <- X[nn_indices[sample_idx, ], , drop = FALSE]
      return(get_feature_wise_mode(neighbors))
    }
  )))
}

feature_value_distances <- function(features, y) {
  return(as.matrix(
    dist(
      prop.table(
        table(droplevels(as.factor(features)), y), 1
      ),
      method = "manhattan"
    )
  ))
}

sample_distances <- function(X, y) {
  # X must be a character matrix
  n_samples <- nrow(X)
  if (length(y) != n_samples) stop("data and label not consistent")
  return(Reduce("+", lapply(seq_len(ncol(X)), function(column) {
    features <- X[, column]
    feature_value_distances(features, y)[features, features]^2
  })))
}

knn_from_distance <- function(D, k_neighbors) {
  return(t(apply(D, 1, function(distances_per_sample) {
    order(distances_per_sample)[1:k_neighbors]
  })))
}
