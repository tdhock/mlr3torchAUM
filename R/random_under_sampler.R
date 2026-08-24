RandomUnderSampler <- R6::R6Class(
  "RandomUnderSampler",
  inherit = BaseUnderSampler,
  public = list(
    replacement = FALSE,
    initialize = function(sampling_strategy = "auto", replacement = FALSE) {
      super$initialize(sampling_strategy)
      if (!is.logical(replacement) || length(replacement) != 1L || is.na(replacement)) {
        stop("replacement must be TRUE or FALSE")
      }
      self$replacement <- replacement
    }
  ),
  private = list(
    .fit_resample = function(X, y) {
      target_strategy <- self$sampling_strategy_

      indices_by_class <- split(seq_along(y), y)
      
      kept_indices_list <- lapply(names(indices_by_class), function(class_name) {
        class_idx <- indices_by_class[[class_name]]
        n_samples <- length(class_idx)
        if (n_samples == 0L) return(integer(0))
        
        if (class_name %in% names(target_strategy)) {
          n_target <- target_strategy[[class_name]]
          class_idx[sample.int(n_samples, n_target, replace = self$replacement)]
        } else {
          class_idx
        }
      })
      
      kept_indices <- unlist(kept_indices_list, use.names = FALSE)

      new_X <- X[kept_indices, , drop = FALSE]
      new_y <- y[kept_indices]

      return(list(X = new_X, y = new_y))
    }
  )
)
