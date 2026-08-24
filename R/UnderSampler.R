BaseUnderSampler <- R6::R6Class(
  "BaseUnderSampler",
  inherit = BaseSampler,
  public = list(
    sampling_type = "under-sampling"
  )
)

