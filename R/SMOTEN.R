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
