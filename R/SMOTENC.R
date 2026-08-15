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

make_samples_nc <- function(continuous, categorical, nn_idx, n_to_generate) {
  n_samples <- nrow(continuous)
  if (n_samples != nrow(categorical) || n_samples != nrow(nn_idx)) stop("data dimension not consistent")
  if (n_to_generate < 1) {
    return(
      list(
        continuous = continuous[integer(0), , drop = FALSE],
        categorical = categorical[integer(0), , drop = FALSE]
      )
    )
  }
  rows <- sample.int(n_samples, n_to_generate, replace = TRUE)
  cols <- sample.int(ncol(nn_idx), n_to_generate, replace = TRUE)
  steps <- runif(n_to_generate, 0, 1)
  generated_cont <- generate_samples(continuous, nn_idx, rows, cols, steps)
  generated_cat <- do.call(rbind, lapply(rows, function(row_num) {
    neighbors <- categorical[nn_idx[row_num, ], , drop = FALSE]
    get_feature_wise_mode(neighbors, tie_break = "random")
  }))
  list(continuous = generated_cont, categorical = generated_cat)
}

SMOTENC <- R6::R6Class(
  "SMOTENC",
  inherit = SMOTE,
  public = list(
    categorical_features = NULL,
    initialize = function(sampling_strategy = "auto", k_neighbors = 5, categorical_features = "auto") {
      super$initialize(sampling_strategy = sampling_strategy, k_neighbors = k_neighbors)
      self$categorical_features <- categorical_features
    }
  ),
  private = list(
    .fit_resample = function(X, y) {
      private$.validate_estimator()
      splitted_features <- split_features(X, self$categorical_features)
      results <- lapply(names(self$sampling_strategy_), function(class_name) {
        n_to_generate <- self$sampling_strategy_[[class_name]]
        in_class_indices <- as.character(y) == class_name
        X_within_class_cont <- splitted_features$continuous[in_class_indices, , drop = FALSE]
        X_within_class_cat <- splitted_features$categorical[in_class_indices, , drop = FALSE]
        nn_idx <- knn_from_distance(smotenc_distances(X_within_class_cont, X_within_class_cat), self$k_neighbors + 1)[, -1, drop = FALSE]
        generated <- make_samples_nc(X_within_class_cont, X_within_class_cat, nn_idx, n_to_generate)
        return(list(
          X = data.frame(generated$continuous, generated$categorical)[, colnames(X)], # assembly and reorder the cols
          y = rep(class_name, n_to_generate)
        ))
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
