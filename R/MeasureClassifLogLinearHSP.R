LogLinearHSP <- function(pred_tensor, label_tensor, margin){
    N <- NULL
    ## Above to avoid CRAN NOTE.
    if(margin < 0)stop("margin must be non-negative")
    is_positive = label_tensor$flatten() == 1
    is_negative = !is_positive
    if(all(as.logical(is_positive)) || all(as.logical(is_negative))){
        return(torch::torch_sum(pred_tensor*0))
    }
    pos_const = torch::torch_where(is_positive, 1, 0)
    neg_const = torch::torch_where(is_positive, 0, 1)
    pos_lin = pos_const*pred_tensor
    pos_m = pos_const*margin
    secondary_indices<-torch::torch_sort(pos_const, stable = TRUE)[[2]]
    reordered_tensor<-(pred_tensor+neg_const*margin)[secondary_indices]
    primary_indices<-torch::torch_sort(reordered_tensor,stable=TRUE)[[2]]
    sorted_indices<-secondary_indices[primary_indices]
    coeff_A = pos_const[sorted_indices]$cumsum(dim=1)
    coeff_B = (2*(pos_m-pos_lin))[sorted_indices]$cumsum(dim=1)
    coeff_C = ((pos_m-pos_lin)^2)[sorted_indices]$cumsum(dim=1)
    neg_lin_s = (neg_const*pred_tensor)[sorted_indices]
    neg_sqr_s = (neg_const*pred_tensor^2)[sorted_indices]
    neg_const_s = neg_const[sorted_indices]
    return(torch::torch_sum(coeff_A*neg_sqr_s+coeff_B*neg_lin_s+coeff_C*neg_const_s))
}
