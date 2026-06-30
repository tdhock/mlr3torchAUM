count_class <- function(y) {
  table_y <- table(y)
  return(setNames(as.integer(table_y), names(table_y)))
}
