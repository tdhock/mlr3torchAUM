is_ce_step <- function(step, k) {
  return(step %% (2 * k) < k)
}
