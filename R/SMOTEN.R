get_feature_wise_mode <- function(neighbors) {
  return(apply(neighbors, 2, function(column) {
    counts <- table(column)
    names(counts)[which.max(counts)]
  }))
}
