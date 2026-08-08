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

dual_batch_list <- function(pos_pool, neg_pool, num_pos, num_neg, num_batches) {
  res <- vector("list", num_batches)
  ptr_pos <- 0
  ptr_neg <- 0
  for (i in seq_len(num_batches)) {
    res_pos <- dual_take(pos_pool, ptr_pos, num_pos)
    res_neg <- dual_take(neg_pool, ptr_neg, num_neg)
    res[[i]] <- c(res_pos$taken, res_neg$taken)
    pos_pool <- res_pos$pool
    neg_pool <- res_neg$pool
    ptr_pos <- res_pos$ptr
    ptr_neg <- res_neg$ptr
  }
  res
}
