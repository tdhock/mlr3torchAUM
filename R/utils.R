get_shuffled_index <- function(N, shuffle) {
  if (shuffle) {
    if (torch::torch_is_installed()) {
      # Use torch for faster permutation if available
      torch::as_array(torch::torch_randperm(N)) + 1L
    } else {
      # Fallback to base R
      sample(N)
    }
  } else {
    seq_len(N)
  }
}