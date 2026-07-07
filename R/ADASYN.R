BaseADASYN <- R6::R6Class(
  "BaseADASYN",
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
      self$nn_k_ <- function(X_within_class) {
        knn_index(X_within_class, X_within_class, self$k_neighbors)
      }
    },
    .make_samples = function(X, nn, n_samples_generate) {
      n_samples_actual <- sum(n_samples_generate)
      rows <- rep(seq_len(nrow(X)), n_samples_generate)
      cols <- sample.int(ncol(nn), n_samples_actual, replace = TRUE)
      steps <- runif(n_samples_actual, 0, 1)
      return(generate_samples(X, nn, rows, cols, steps, nn_data = X))
    }
  )
)

ADASYN <- R6::R6Class(
  "ADASYN",
  inherit = BaseADASYN,
  private = list(
    .fit_resample = function(X, y) {
      private$.validate_estimator()
      results <- lapply(names(self$sampling_strategy_), function(class_name) {
        n_samples <- self$sampling_strategy_[[class_name]]
        if (n_samples == 0) return(NULL)

        class_indices <- as.character(y) == class_name
        X_class <- X[class_indices, , drop = FALSE]
        knn_idx <- knn_index(query = X_class, data = X, k = self$k_neighbors)

        is_not_class <- matrix(as.character(y)[knn_idx] != class_name, nrow = nrow(knn_idx))
        ratio_nn <- rowSums(is_not_class) / self$k_neighbors
        sum_ratio <- sum(ratio_nn)
        if (sum_ratio == 0) {
          stop("Not any neighbors belong to the majority class. ADASYN is not suited for this specific dataset. Use SMOTE instead.")
        }
        ratio_nn <- ratio_nn / sum_ratio
        n_samples_generate <- round(ratio_nn * n_samples)
        n_samples_actual <- sum(n_samples_generate)
        if (n_samples_actual == 0) {
          stop("No samples will be generated with the provided ratio settings.")
        }
        nn_idx_within <- self$nn_k_(X_class)
        generated <- private$.make_samples(X_class, nn_idx_within, n_samples_generate)
        return(list(X = generated, y = rep(class_name, n_samples_actual)))
      })
      results <- results[!sapply(results, is.null)]
      new_X <- do.call(rbind, c(list(X), lapply(results, function(result) result$X)))
      new_y <- factor(c(as.character(y), unlist(lapply(results, function(result) result$y))),
        levels = levels(y)
      )
      return(list(X = new_X, y = new_y))
    }
  )
)
