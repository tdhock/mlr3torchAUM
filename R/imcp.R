trapz <- function(x, y) {
    sum((y[-1] + y[-length(y)]) * diff(x) / 2)
}
