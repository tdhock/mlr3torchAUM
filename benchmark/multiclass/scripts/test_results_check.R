csv_summary<-function(csv_file){
    repo.dir<-system("git rev-parse --show-toplevel", intern = TRUE)
    csv_file<-file.path(repo.dir, "benchmark", "multiclass", "data", csv_file)
    d<-read.csv(csv_file)
    cat("File to check: ", csv_file, "\n")
    cat("1. Number of rows: ", nrow(d), "\n")
    cat("2. Number of Classes: ", length(unique(d$y_true)), "\n")
    cat("3. Class distribution: ", table(d$y_true), "\n")
    dist.pct<-table(d$y_true)/nrow(d)*100
    cat("4. Class distribution percentage: ", sprintf("%.2f%%", dist.pct), "\n")
}

csv_summary("test_imbalanced_class_probs.csv")
csv_summary("test_results.csv")