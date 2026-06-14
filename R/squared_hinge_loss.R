SquaredHingeLoss <- function(pred_tensor, label_tensor, margin=1){
  if(length(margin) != 1 || margin < 0){
    stop("margin must be a non-negative scalar")
  }
  yhat = pred_tensor$flatten()
  y = label_tensor$flatten()
  is_positive = y == 1
  is_negative = !is_positive
  v = yhat + is_negative$to(dtype=yhat$dtype) * margin
  s = torch::torch_argsort(v)
  yhat_s = yhat[s]
  is_positive_s = is_positive[s]
  I_pos = torch::torch_where(is_positive_s, 1, 0)$to(dtype=yhat_s$dtype)
  I_neg = 1 - I_pos
  z = (margin - yhat_s) * I_pos
  a = I_pos$cumsum(dim=1)
  b = (2 * z)$cumsum(dim=1)
  c = (z^2)$cumsum(dim=1)
  L_i = (a * yhat_s^2 + b * yhat_s + c) * I_neg
  torch::torch_sum(L_i)
}

nn_squared_hinge_loss <- torch::nn_module(
  c("nn_squared_hinge_loss", "nn_loss"),
  initialize = function(margin=1) {
    self$margin = margin
    for(name in c("evals","zeros","all_one_class")){
      self$buffer(name)
    }
  },
  buffer = function(name, value=torch::torch_tensor(0L)){
    self[[name]] <- torch::nn_buffer(value)
  },
  increment = function(name)self$buffer(name, self[[name]]+1L),
  forward = function(pred_tensor, label_tensor){
    loss_tensor <- SquaredHingeLoss(pred_tensor, label_tensor, self$margin)
    self$increment("evals")
    if(torch::as_array(loss_tensor==0))self$increment("zeros")
    if(torch::as_array((label_tensor[1]==label_tensor)$all()))
      self$increment("all_one_class")
    loss_tensor
  }
)

torch_loss_sq_hinge_loglinear <- function() {
  mlr3torch::TorchLoss$new(
    torch_loss = nn_squared_hinge_loss,
    task_types = "classif",
    id = "sq_hinge_loglinear",
    label = "Log-linear squared hinge loss",
    packages = "mlr3torchAUM",
    man = "mlr3torchAUM::nn_squared_hinge_loss"
  )
}
