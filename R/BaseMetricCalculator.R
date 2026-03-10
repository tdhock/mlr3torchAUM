BaseMetricCalculator <- function(.label_sanity_check=NULL){
    if(is.null(.label_sanity_check)){
        .label_sanity_check = function(is_positive){
            return(is_positive)}}
    return(function(pred_tensor, label_tensor){
        .peek_label(label_tensor) |>
        .label_sanity_check() |>
        .compute_basic_metrics(pred_tensor, label_tensor)
    })}

.compute_basic_metrics <- function(is_positive, pred_tensor, label_tensor){
    is_negative = !is_positive
    fn_diff = torch::torch_where(is_positive, -1, 0)
    fp_diff = torch::torch_where(is_positive, 0, 1)
    thresh_tensor = -pred_tensor$flatten()
    sorted_indices = torch::torch_argsort(thresh_tensor)
    p_total= torch::torch_sum(is_negative)
    n_total = torch::torch_sum(is_positive)
    sorted_fp_cum = fp_diff[sorted_indices]$cumsum(dim=1)/n_total
    sorted_fn_cum = -fn_diff[sorted_indices]$flip(dims=1)$cumsum(dim=1)$flip(dims=1)/p_total
    sorted_thresh = thresh_tensor[sorted_indices]
    sorted_is_diff = sorted_thresh$diff() != 0
    sorted_fp_end = torch::torch_cat(list(sorted_is_diff, torch::torch_tensor(TRUE)))
    sorted_fn_end = torch::torch_cat(list(torch::torch_tensor(TRUE), sorted_is_diff))
    uniq_thresh = sorted_thresh[sorted_fp_end]
    uniq_fp_after = sorted_fp_cum[sorted_fp_end]
    uniq_fn_before = sorted_fn_cum[sorted_fn_end]
    FPR = torch::torch_cat(list(torch::torch_tensor(0.0), uniq_fp_after))
    FNR = torch::torch_cat(list(uniq_fn_before, torch::torch_tensor(0.0)))
    return(list(p_total=p_total, n_total=n_total,
    FPR=FPR, FNR=FNR,
    min_constant=torch::torch_cat(list(torch::torch_tensor(-Inf), uniq_thresh)),
    max_constant=torch::torch_cat(list(uniq_thresh, torch::torch_tensor(Inf)))))
}

