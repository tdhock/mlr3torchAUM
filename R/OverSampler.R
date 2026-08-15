BaseOverSampler <- R6::R6Class(
  "BaseOverSampler",
  inherit = BaseSampler,
  public = list(
    sampling_type = "over-sampling"
  )
)
