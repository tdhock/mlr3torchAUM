AUCMLoss <- function(pred_tensor, label_tensor) {
    device <- pred_tensor$device
    return(torch::torch_zeros(1, device=device))
}