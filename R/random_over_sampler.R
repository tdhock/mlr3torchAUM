RandomOverSampler <- R6::R6Class(
  "RandomOverSampler",
  inherit = BaseOverSampler,
  public = list(
    shrinkage = NULL,
    initialize = function(sampling_strategy = "auto", shrinkage = NULL) {
      super$initialize(sampling_strategy)
      if (!is.null(shrinkage) && (!is.numeric(shrinkage) || length(shrinkage) != 1L || shrinkage < 0)) {
        stop("shrinkage must be a non-negative number")
      }
      self$shrinkage <- shrinkage
    }
  ),
  private = list(
    .fit_resample = function(X, y) {
      results <- lapply(names(self$sampling_strategy_), function(class_name) {
        n_to_generate <- self$sampling_strategy_[[class_name]]
        if (n_to_generate == 0L) return(NULL)

        class_idx <- which(y == class_name)
        n_samples <- length(class_idx)
        if (n_samples == 0L) return(NULL)

        rows <- class_idx[sample.int(n_samples, n_to_generate, replace = TRUE)]
        X_resampled <- X[rows, , drop = FALSE]

        if (!is.null(self$shrinkage) && self$shrinkage > 0) {
          n_features <- ncol(X)
          h <- (4 / ((n_features + 2) * n_samples))^(1 / (n_features + 4))
          stds <- apply(X[class_idx, , drop = FALSE], 2, stats::sd)
          stds[is.na(stds) | stds == 0] <- 0

          noise <- matrix(rnorm(n_to_generate * n_features), nrow = n_to_generate, ncol = n_features)
          scale_vec <- self$shrinkage * h * stds
          X_resampled <- X_resampled + sweep(noise, 2, scale_vec, "*")
        }

        return(list(X = X_resampled, y = rep(class_name, n_to_generate)))
      })

      results <- results[!vapply(results, is.null, logical(1))]

      new_X <- do.call(rbind, c(list(X), lapply(results, function(res) res$X)))
      new_y <- factor(
        c(as.character(y), unlist(lapply(results, function(res) res$y))),
        levels = levels(y)
      )

      return(list(X = new_X, y = new_y))
    }
  )
)
