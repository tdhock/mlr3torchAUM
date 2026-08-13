BaseUnderSampler <- R6::R6Class(
  "BaseUnderSampler",
  inherit = BaseSampler,
  public = list(
    sampling_type = "under-sampling",
    fit_resample = function(X, y) {
      if (nrow(X) != length(y)) stop("data and label not consistent")
      sampling_strategy <- check_sampling_strategy_under(y, self$sampling_strategy)
      self$sampling_strategy_ <- sampling_strategy[sampling_strategy > 0]
      private$.fit_resample(X, y)
    }
  )
)
