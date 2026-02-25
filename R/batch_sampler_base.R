# Base sampler shared by batch samplers.
# Provides common .iter and .length used by both RandomSampler and StratifiedSampler.
BaseBatchSampler <- R6::R6Class(
  "BaseBatchSampler",
  public = list(
    batch_list = NULL,
    .iter = function() {
      batch.i <- 0
      function() {
        if (batch.i < length(self$batch_list)) {
          batch.i <<- batch.i + 1L
          indices <- self$batch_list[[batch.i]]
          if (batch.i == length(self$batch_list)) {
            self$set_batch_list()
          }
          return(indices)
        }
        coro::exhausted()
      }
    },
    .length = function() {
      length(self$batch_list)
    }
  )
)