BaseSampler <- R6::R6Class(
  "BaseSampler",
  public = list(
    sampling_strategy = NULL,
    initialize = function(sampling_strategy = "auto") self$sampling_strategy <- sampling_strategy,
    fit_resample = function(X, y) private$.fit_resample(X, y)
  ),
  private = list(
    .fit_resample = function(X, y) stop("not implemented")
  )
)
