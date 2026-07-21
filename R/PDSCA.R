pdsca_step <- function(
  loss_module, lr, clamp_value, weight_decay,
  epoch_decay, momentum, pass, state_a = NULL, state_b = NULL
) {
  if (pass == "aucm") {
    res <- pesg_full_step(
      loss_module,
      lr, clamp_value, weight_decay,
      epoch_decay, momentum, state_a, state_b
    )
    state_a <- res$state_a
    state_b <- res$state_b
  }
  return(list(state_a = state_a, state_b = state_b))
}

pdsca_pass <- function(loss_fn){
  if(is_ce_step(loss_fn$step$item()-1L, loss_fn$k)) "ce" else "aucm"
}