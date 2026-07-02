BaseSampler <- R6::R6Class(
  "BaseSampler",
  public = list(
    sample_strategy = NULL,
    initialize = function(sample_strategy = "auto") self$sample_strategy <- sample_strategy,
    fit_resample = function(X, y) private$.fit_resample(X, y)
  ),
  private = list(
    .fit_resample = function(X, y) stop("not implemented")
  )
)
