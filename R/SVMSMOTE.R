minority_support_vectors <- function(svm_index, y, target_class) {
  return(svm_index[as.character(y[svm_index]) == target_class])
}
