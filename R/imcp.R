trapz <- function(x, y) {
    sum((y[-1] + y[-length(y)]) * diff(x) / 2)
}

get_y_values <- function(y_true, y_true_score, y_score) {
    hellinger_distance <- sqrt(
        rowSums((y_true_score - sqrt(y_score))^2))/sqrt(2)
    curve_y <- 1 - hellinger_distance
    ord <- order(curve_y, y_true)
    return(list(curve_y = curve_y[ord], sort_indices = ord))
}

map_class_labels <- function(y_true, y_score, labels = NULL) {
    unique_classes <- sort(unique(y_true))
    return(list(y_true_size = length(unique_classes),
    y_true_int_encoded = match(y_true, unique_classes)))
}