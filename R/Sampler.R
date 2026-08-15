BaseSampler <- R6::R6Class(
  "BaseSampler",
  public = list(
    sampling_strategy = NULL,
    sampling_strategy_ = NULL,
    initialize = function(sampling_strategy = "auto") self$sampling_strategy <- sampling_strategy,
    fit_resample = function(X, y) {
      if (nrow(X) != length(y)) stop("data and label not consistent")
      sampling_strategy <- check_sampling_strategy(y, self$sampling_strategy)
      self$sampling_strategy_ <- sampling_strategy[sampling_strategy > 0]
      private$.fit_resample(X, y)
    }
  ),
  private = list(
    .fit_resample = function(X, y) stop("not implemented")
  )
)
