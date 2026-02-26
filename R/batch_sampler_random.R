batch_sampler_random <- function(batch_size, shuffle=TRUE){
  .N <- `:=` <- i.in.stratum <- . <- max.i <- n.samp <- batch.i <- self <- NULL
  ## Above for CRAN check.
  methods <- batch_sampler_methods(shuffle)
  torch::sampler(
    "RandomSampler",
    initialize = function(data_source) {
      self$N <- data_source$task$nrow
      self$batch_vec <- seq_len(self$N) %/% batch_size
      self$set_batch_list()
    },
    set_batch_list = function() {
      index_vec <- self$.shuffled_index(self$N)
      self$batch_list <- split(index_vec, self$batch_vec)
    },
    .shuffled_index = methods$.shuffled_index,
    .iter = methods$.iter,
    .length = methods$.length
  )
}
