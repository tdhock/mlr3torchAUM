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
