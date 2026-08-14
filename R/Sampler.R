count_class <- function(y) {
  table_y <- table(y)
  table_y <- table_y[table_y > 0]
  return(setNames(as.integer(table_y), names(table_y)))
}

check_sampling_strategy <- function(y, strategy = "auto", sampling_type = "over-sampling") {
  count <- count_class(y)
  if (sampling_type == "over-sampling") {
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
  } else if (sampling_type == "under-sampling") {
    min_value <- min(count)
    if (strategy == "auto") {
      other_count <- count[count > min_value]
      return(setNames(rep.int(min_value, length(other_count)), names(other_count)))
    }
    if (strategy == "all") {
      return(setNames(rep.int(min_value, length(count)), names(count)))
    }
  }
  stop(sprintf("strategy %s not implemented for %s", strategy, sampling_type))
}

BaseSampler <- R6::R6Class(
  "BaseSampler",
  public = list(
    sampling_type = "over-sampling",
    sampling_strategy = NULL,
    sampling_strategy_ = NULL,
    initialize = function(sampling_strategy = "auto") {
      self$sampling_strategy <- sampling_strategy
    },
    fit_resample = function(X, y) {
      if (nrow(X) != length(y)) stop("data and label not consistent")
      sampling_strategy <- check_sampling_strategy(y, self$sampling_strategy, self$sampling_type)
      self$sampling_strategy_ <- sampling_strategy[sampling_strategy > 0]
      private$.fit_resample(X, y)
    }
  ),
  private = list(
    .fit_resample = function(X, y) stop("not implemented")
  )
)
