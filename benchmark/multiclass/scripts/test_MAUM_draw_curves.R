# This script tests R/Draw_ROC_macro.R/Draw_ROC_curve_macro 
# and R/Draw_ROC_micro.R/Draw_ROC_curve_micro functions
#
# The test data are described in https://github.com/mlr3-imbalanced/mlr3torchAUM/issues/40
# "Predicted probability format (y_true + y_score_1..K, sum of probabilities per row = 1)"
# located in benchmark/multiclass/data/test_results.csv 
# and benchmark/multiclass/data/test_imbalanced_class_probs.csv.
#
# To run this script, run
# Rscript path/to/this/script/test_MAUM_draw_curves.R

library(testthat)
library(data.table)
data.table::setDTthreads(1L)
library(ggplot2)
torch_available <- requireNamespace("torch") && torch::torch_is_installed()
stopifnot(torch_available)

repo.dir <- system("git rev-parse --show-toplevel", intern = TRUE)
for(test.file in c("Draw_ROC_micro.R", "Draw_ROC_macro.R")){
  source(file.path(repo.dir, "R", test.file))
}
data.dir <- file.path(repo.dir, "benchmark", "multiclass", "data")
fig.dir <- file.path(repo.dir, "benchmark", "multiclass", "figures")
dir.create(fig.dir, showWarnings = FALSE)

read_probs <- function(data.file){
  data.dt <- fread(file.path(data.dir, data.file))
  y_true <- as.integer(data.dt$y_true)
  if(min(y_true) == 0L) y_true <- y_true + 1L # change 0-based to 1-based
  prob_mat <- as.matrix(data.dt[, !"y_true"])
  list(y_true = y_true,
  prob_mat = prob_mat,
  pred_tensor = torch::torch_tensor(prob_mat),
  label_tensor = torch::torch_tensor(y_true))
}

data_list <- list()
# Draw the pictures and save to fig.dir
for (data.file in c("test_results.csv", "test_imbalanced_class_probs.csv")) {
    probs <- read_probs(data.file)
    data_list[[data.file]] <- probs
    gg.micro <- Draw_ROC_curve_micro(probs$pred_tensor, probs$label_tensor)
    gg.macro <- Draw_ROC_curve_macro(probs$pred_tensor, probs$label_tensor)
    ggsave(file.path(fig.dir, paste0(data.file, "_ROC_micro.png")),
        gg.micro, width = 7, height = 5, dpi = 120)
    ggsave(file.path(fig.dir, paste0(data.file, "_ROC_macro.png")),
        gg.macro, width = 7, height = 5, dpi = 120)
}

sessionInfo() # need after plotting

trapezoid_auc <- function(fpr, tpr) sum(diff(fpr)*(tpr[-1]+tpr[-length(fpr)])/2)

test_that("test_imbalanced_class_probs matches expected data shape", {
    probs <- data_list[["test_imbalanced_class_probs.csv"]]
    prob_mat <- probs$prob_mat
    expect_equal(ncol(prob_mat), 3L)
    expect_equal(as.integer(table(probs$y_true)), c(100L, 1600L, 8100L))
    expect_equal(rowSums(prob_mat),
    rep(1, length(probs$y_true)), tolerance = 1e-6)
})

test_that("micro ROC curves match its mathematical properties", {
    probs <- data_list[["test_imbalanced_class_probs.csv"]]
    gg <- Draw_ROC_curve_micro(probs$pred_tensor, probs$label_tensor)
    expect_s3_class(gg, "ggplot")
    curve <- data.table(gg$data)
    expect_true(all(curve$FPR >= 0 & curve$FPR <= 1))
    expect_equal(curve[1, c(FPR, TPR)], c(0, 0))
    expect_equal(curve[.N, c(FPR, TPR)], c(1, 1))
    expect_true(all(diff(curve$FPR) >= 0))
    expect_true(all(diff(curve$TPR) >= 0))
})

test_that("micro AUC agrees with mlr3measures::auc",{
    probs <- data_list[["test_imbalanced_class_probs.csv"]]
    gg <- Draw_ROC_curve_micro(probs$pred_tensor, probs$label_tensor)
    curve <- data.table(gg$data)
    area_curve <- trapezoid_auc(curve$FPR, curve$TPR)
    one_hot_flat <- as.integer(col(probs$prob_mat)==probs$y_true)
    score_flat <- as.numeric(probs$prob_mat)
    area_ref <- mlr3measures::auc(factor(one_hot_flat, levels = c(0, 1)),
    score_flat, positive = "1")
    expect_equal(area_curve, area_ref, tolerance = 1e-8)
})

test_that("macro AUC agrees with mlr3measures::auc on test_results.csv", {
    probs <- data_list[["test_results.csv"]] # rare ties
    gg <- Draw_ROC_curve_macro(probs$pred_tensor, probs$label_tensor)
    curve <- data.table(gg$data) # fpr tpr class
    expect_equal(length(unique(curve$class)), 7L)
    auc_by_class <- curve[, .(auc=trapezoid_auc(fpr,tpr)), by=class]
    reference <- sapply(1:7, function(k){
        mlr3measures::auc(factor(probs$y_true==k, levels=c(FALSE, TRUE)),
        probs$prob_mat[, k], positive="TRUE")
    })
    expect_equal(auc_by_class$auc, reference, tolerance = 1e-8)
})

test_that("macro AUC agrees with mlr3measures::auc on test_imbalanced_class_probs.csv", {
    probs <- data_list[["test_imbalanced_class_probs.csv"]] # mostly ties
    gg <- Draw_ROC_curve_macro(probs$pred_tensor, probs$label_tensor)
    curve <- data.table(gg$data) # fpr tpr class
    expect_equal(length(unique(curve$class)), 3L)
    auc_by_class <- curve[, .(auc=trapezoid_auc(fpr,tpr)), by=class]
    reference <- sapply(1:3, function(k){ # one vs. all
        mlr3measures::auc(factor(probs$y_true==k, levels=c(FALSE, TRUE)),
        probs$prob_mat[, k], positive="TRUE")
    })
    expect_equal(auc_by_class$auc, reference, tolerance = 0.01) # pass
    
    # see https://github.com/mlr3-imbalanced/mlr3torchAUM/issues/47
    expect_equal(auc_by_class$auc, reference, tolerance = 1e-8) # fail
})
