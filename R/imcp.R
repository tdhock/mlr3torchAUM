trapz <- function(x, y) {
    sum((y[-1] + y[-length(y)]) * diff(x) / 2)
}

get_y_values <- function(y_true, y_true_score_one_hot, y_score) {
    hellinger_distance <- sqrt(
        rowSums((y_true_score_one_hot - sqrt(y_score))^2))/sqrt(2)
    curve_y <- 1 - hellinger_distance
    ord <- order(curve_y, y_true)
    return(list(curve_y = curve_y[ord], sort_indices = ord))
}

map_class_labels <- function(y_true, y_score, labels = NULL) {
    unique_classes <- sort(unique(y_true))
    y_true_size <- length(unique_classes)
    y_score_size <- ncol(y_score)
    if (y_true_size != y_score_size) {
        if (is.null(labels)) stop("labels not given")
        if (length(labels) != y_score_size) stop(
            "labels length not equal to y_score columns number")
        if (class(labels) != class(y_true)) stop(
            "labels type does not match y_true type")
        if( !all(unique_classes %in% labels)) stop(
            "y_true classes are not a subset of labels")
        unique_classes <- sort(labels)
        y_true_size <- y_score_size
    }
    return(list(y_true_size = y_true_size,
    y_true_int_encoded = match(y_true, unique_classes)))
}

prepare_curve <- function(y_true, y_score, labels, abs_tolerance) {
    n_samples <- nrow(y_score)
    if (n_samples != length(y_true)) stop("sample numbers do not match")
    if (any(abs(rowSums(y_score) - 1) > abs_tolerance)) stop(
        "Not all rows' probabilities sum to 1")
    class_label_mapping <- map_class_labels(y_true, y_score, labels)
    y_true_score_one_hot <- diag(class_label_mapping$y_true_size)[
        class_label_mapping$y_true_int_encoded,]
    y_values <- get_y_values(y_true, y_true_score_one_hot, y_score)
    return(list(n_samples=n_samples,
    class_label_mapping=class_label_mapping,
    y_true_score_one_hot=y_true_score_one_hot,
    curve_y=y_values$curve_y,
    sort_indices=y_values$sort_indices))
}

mcp_curve <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8) {
    prepared <- prepare_curve(y_true, y_score, labels, abs_tolerance)
    n_samples <- prepared$n_samples
    curve_y <- prepared$curve_y
    return(list(curve_x = (0:(n_samples-1)) / (n_samples-1),
    curve_y = curve_y))
}

mcp_score <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8) {
    cur <- mcp_curve(y_true, y_score, labels, abs_tolerance)
    return(trapz(cur$curve_x, cur$curve_y))
}

get_class_widths <- function(y_true_score_one_hot, y_true_size) {
    counts <- colSums(y_true_score_one_hot)
    return(1 / (y_true_size * counts))
}

imcp_curve <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8) {
    prepared <- prepare_curve(y_true, y_score, labels, abs_tolerance)
    y_true_score_one_hot <- prepared$y_true_score_one_hot
    class_label_mapping <- prepared$class_label_mapping
    sort_indices <- prepared$sort_indices
    curve_y <- prepared$curve_y
    class_widths <- get_class_widths(y_true_score_one_hot,
    class_label_mapping$y_true_size)
    sample_widths <- class_widths[class_label_mapping$y_true_int_encoded]
    curve_x <- sample_widths[sort_indices]
    curve_x <- cumsum(curve_x) - curve_x / 2
    curve_x <- c(0, curve_x, 1)
    curve_y <- c(curve_y[1], curve_y, curve_y[length(curve_y)])
    return(list(curve_x = curve_x, curve_y = curve_y))
}

imcp_score <- function(y_true, y_score, labels = NULL, abs_tolerance = 1e-8) {
    cur <- imcp_curve(y_true, y_score, labels, abs_tolerance)
    return(trapz(cur$curve_x, cur$curve_y))
}