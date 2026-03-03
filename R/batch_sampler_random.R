batch_sampler_random <- function(batch_size, shuffle=TRUE){
  .N <- `:=` <- i.in.stratum <- . <- max.i <- n.samp <- batch.i <- self <- NULL
  ## Above for CRAN check.
  torch::sampler(
    "RandomSampler",
    inherit = batch_sampler_base,
    initialize = function(data_source) {
      self$N <- data_source$task$nrow
      self$batch_vec <- seq_len(self$N) %/% batch_size
      super$initialize(data_source)
    },
    set_batch_list = function() {
      self$batch_list <- split(
        self$shuffle_index(self$N, shuffle),
        self$batch_vec)
    }
  )
}
