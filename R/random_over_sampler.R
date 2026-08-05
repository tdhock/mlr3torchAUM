RandomOverSampler <- R6::R6Class(
  "RandomOverSampler",
  inherit = BaseOverSampler,
  public = list(
    initialize = function(sampling_strategy = "auto") {
      super$initialize(sampling_strategy)
    }
  ),
  private = list(
    .fit_resample = function(X, y) {
      results <- lapply(names(self$sampling_strategy_), function(class_name) {
        n_to_generate <- self$sampling_strategy_[[class_name]]
        if (n_to_generate == 0L) return(NULL)

        X_within_class <- X[as.character(y) == class_name, , drop = FALSE]
        n_samples <- nrow(X_within_class)
        if (n_samples == 0L) return(NULL)

        rows <- sample.int(n_samples, n_to_generate, replace = TRUE)
        X_resampled <- X_within_class[rows, , drop = FALSE]

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
