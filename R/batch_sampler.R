#' @title Base Batch Sampler Methods
#'
#' @name batch_sampler_methods
#'
#' @description
#' Returns a named list of shared methods for use in batch samplers built with
#' `torch::sampler()`. Encapsulates shuffle-aware index generation, iteration,
#' and length logic that is common across all batch sampler variants.
#'
#' Child samplers should call `batch_sampler_methods(shuffle)` and splice the
#' result into their `torch::sampler()` call alongside their own `initialize`
#' and `set_batch_list` methods (see *Inheriting*).
#'
#' @section Inheriting:
#' Concrete samplers must implement:
#' * `initialize(data_source)` :: Set up sampler state and call `self$set_batch_list()`.
#' * `set_batch_list()` :: Populate `self$batch_list` using `self$.shuffled_index(n)`.
#'
#' The following methods are provided by `batch_sampler_methods()` and should
#' not be re-implemented in child samplers:
#' * `.shuffled_index(n)` :: Returns a shuffled or sequential index vector of length `n`.
#' * `.iter()` :: Iterates over `self$batch_list`, re-shuffling at exhaustion.
#' * `.length()` :: Returns the number of batches.
#'
#' @param shuffle (`logical(1)`)\cr
#'   Whether to shuffle indices each epoch. Default is `TRUE`.
#'
#' @return A named `list` of functions to be spliced into `torch::sampler()`.
#'
#' @keywords internal
batch_sampler_methods = function(shuffle = TRUE) {
  list(
    .shuffled_index = function(n) {
      if (shuffle) {
        if (torch::torch_is_installed()) {
          torch::as_array(torch::torch_randperm(n)) + 1L
        } else {
          sample(n)
        }
      } else {
        seq_len(n)
      }
    },
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
}
