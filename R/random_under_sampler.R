check_sampling_strategy_under <- function(y, strategy = "auto") {
  count <- count_class(y)
  min_value <- min(count)

  if (strategy == "auto") {
    maj_counts <- count[count > min_value]
    target_counts <- setNames(rep.int(min_value, length(maj_counts)), names(maj_counts))
    return(target_counts)
  }
  if (strategy == "all") {
    target_counts <- setNames(rep.int(min_value, length(count)), names(count))
    return(target_counts)
  }
  stop(sprintf("strategy %s not implemented for under-sampling", strategy))
}

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
