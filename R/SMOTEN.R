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

SMOTEN <- R6::R6Class("SMOTEN",
  inherit = SMOTE,
  private = list(
    .fit_resample = function(X, y) {
      private$.validate_estimator()
      D_full <- sample_distances(X, y)
      results <- lapply(names(self$sampling_strategy_), function(class_name) {
        n_to_generate <- self$sampling_strategy_[[class_name]]
        within_class_idx <- which(as.character(y) == class_name)
        D_within_class <- D_full[within_class_idx, within_class_idx, drop = FALSE]
        nn <- knn_from_distance(D_within_class, self$k_neighbors + 1)
        X_within_class <- X[within_class_idx, , drop = FALSE]
        generated <- make_samples_nominal(X_within_class, nn, n_to_generate)
        list(X = generated, y = rep(class_name, n_to_generate))
      })
      list(
        X = do.call(rbind, c(list(X), lapply(results, function(result) result$X))),
        y = factor(c(as.character(y), unlist(lapply(results, function(result) result$y))),
          levels = levels(y)
        )
      )
    }
  )
)
