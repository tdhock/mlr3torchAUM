dual_num_pos <- function(batch_size, sampling_rate, num_pos = NULL) {
  if (!is.null(sampling_rate) && !is.null(num_pos)) {
    stop("sampling_rate and num_pos cannot be given at same time")
  }
  if (is.null(sampling_rate)) {
    return(min(batch_size, num_pos))
  }
  max(as.integer(batch_size * sampling_rate), 1L)
}

dual_num_batches <- function(pos_len, neg_len, num_pos, num_neg) {
  max(pos_len %/% num_pos, neg_len %/% num_neg)
}

dual_class_indices <- function(labels) {
  if (!all(labels %in% c(0, 1))) stop("labels must be 0 or 1")
  if (!any(labels == 0L) || !any(labels == 1L)) stop("must have both labels")
  list(neg = which(labels == 0L), pos = which(labels == 1L))
}

dual_take <- function(pool, ptr, need) {
  pool_length <- length(pool)
  if (ptr + need < pool_length) {
    return(list(
      pool = pool, ptr = ptr + need,
      taken = pool[ptr + seq_len(need)]
    ))
  }
  num_loops <- (need - (pool_length - ptr)) %/% pool_length
  new_ptr <- (ptr + need) %% pool_length
  tail_part <- pool[ptr + seq_len(pool_length - ptr)]
  pool <- sample(pool)
  taken <- c(tail_part, rep(pool, num_loops), pool[seq_len(new_ptr)])
  list(pool = pool, ptr = new_ptr, taken = taken)
}

dual_batch_list <- function(
  pos_pool, neg_pool, num_pos, num_neg, num_batches,
  pos_ptr = 0, neg_ptr = 0
) {
  batches <- vector("list", num_batches)
  for (i in seq_len(num_batches)) {
    res_pos <- dual_take(pos_pool, pos_ptr, num_pos)
    res_neg <- dual_take(neg_pool, neg_ptr, num_neg)
    batches[[i]] <- c(res_pos$taken, res_neg$taken)
    pos_pool <- res_pos$pool
    neg_pool <- res_neg$pool
    pos_ptr <- res_pos$ptr
    neg_ptr <- res_neg$ptr
  }
  list(
    batches = batches, pos_pool = pos_pool, neg_pool = neg_pool,
    pos_ptr = pos_ptr, neg_ptr = neg_ptr
  )
}

batch_sampler_dual <- function(batch_size, sampling_rate = 0.5, num_pos = NULL,
                               shuffle = TRUE, random_seed = NULL) {
  self <- NULL
  ## Above for CRAN check.
  torch::sampler(
    "DualSampler",
    initialize = function(data_source) {
      self$data_source <- data_source
      self$batch_size <- batch_size
      TSK <- data_source$task
      labels <- as.integer(TSK$truth() == TSK$positive)
      if (!is.null(random_seed)) set.seed(random_seed)
      self$num_pos <- dual_num_pos(batch_size, sampling_rate, num_pos)
      self$num_neg <- batch_size - self$num_pos
      idx <- dual_class_indices(labels)
      self$pos_pool <- if (shuffle) sample(idx$pos) else idx$pos
      self$neg_pool <- if (shuffle) sample(idx$neg) else idx$neg
      self$num_batches <- dual_num_batches(
        length(self$pos_pool), length(self$neg_pool),
        self$num_pos, self$num_neg
      )
      self$pos_ptr <- 0
      self$neg_ptr <- 0
      self$set_batch_list()
    },
    set_batch_list = function() {
      res <- dual_batch_list(
        self$pos_pool, self$neg_pool, self$num_pos, self$num_neg,
        self$num_batches, self$pos_ptr, self$neg_ptr
      )
      self$batch_list <- res$batches
      self$pos_pool <- res$pos_pool
      self$neg_pool <- res$neg_pool
      self$pos_ptr <- res$pos_ptr
      self$neg_ptr <- res$neg_ptr
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
