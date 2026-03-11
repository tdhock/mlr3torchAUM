NaiveHSP <- function(pred_tensor, label_tensor, margin){
    N <- NULL
    ## Above to avoid CRAN NOTE.
    if(margin < 0)stop("margin must be non-negative")
    is_positive = label_tensor$flatten() == 1
    is_negative = !is_positive
    if(all(as.logical(is_positive)) || all(as.logical(is_negative))){
        return(torch::torch_sum(pred_tensor*0))
    }
    batch_size = label_tensor$flatten()$shape
    neg_const = torch::torch_where(is_positive, 0, 1)
    pos_mat_for_neg = pred_tensor[is_positive]$unsqueeze(2)$`repeat`(
        c(1, batch_size)) * neg_const
    adjusted_neg = (pred_tensor+margin)*neg_const
    diff = torch::torch_relu(adjusted_neg-pos_mat_for_neg)
    return(torch::torch_sum(diff^2))
}
