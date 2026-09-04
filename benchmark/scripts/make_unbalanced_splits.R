library(data.table)

# From https://tdhock.github.io/blog/2026/exact-downsampling/
compute_target_counts <- function(p_neg, Tpos, Tneg) {
  p_small <- ifelse(p_neg < 0.5, p_neg, 1 - p_neg)
  n_small <- as.integer(pmin(
    2 * Tpos * p_small / (3 * (1 - p_neg) + p_neg),
    2 * Tneg * p_small / (1 + 2 * p_neg)))
  data.table(p_neg)[, `:=`(
    n_pos = ifelse(p_neg < 0.5, n_small * (1 - p_neg) / p_neg, n_small),
    n_neg = ifelse(p_neg < 0.5, n_small, n_small * p_neg / (1 - p_neg)))
  ][, n_imb := n_pos + n_neg
  ][, N_pos := n_imb / 2][, N_neg := n_imb - N_pos
  ][, `:=`(pos = n_pos + N_pos, neg = n_neg + N_neg)
  ][, check_prop := n_neg / n_imb][]
}

