.N <- `:=` <- i.in.stratum <- . <- max.i <- n.samp <- batch.i <- self <- NULL
## Above for CRAN check.
BaseSampler <-  torch::sampler(
  "BaseSampler",
  initialize = function(data_source) {
    self$set_batch_list()
  },
  set_batch_list = function() {
    stop("set_batch_list not implemented")
  },
  shuffle_index = function(N, shuffle) {
    if (!shuffle) return(seq_len(N))
    if (torch::torch_is_installed()) {
      return(torch::as_array(torch::torch_randperm(N)) + 1L)
    }
    sample(N)
  },
  .iter = function() {
    batch.i <- 0
    function() {
      if (batch.i < length(self$batch_list)) {
        batch.i <<- batch.i + 1L
        out <- self$batch_list[[batch.i]]
        if (batch.i == length(self$batch_list)) self$set_batch_list()
        return(out)
      }
      coro::exhausted()
    }
  },
  .length = function() length(self$batch_list)
)
