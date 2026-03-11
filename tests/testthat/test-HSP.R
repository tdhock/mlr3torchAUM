library(testthat)
data.table::setDTthreads(1L)

extremely_naive_HSP_loss = function(preds, labels, margin){
    split_samples = split(torch::as_array(preds), 
    torch::as_array(labels))
    neg_samples = split_samples[[1]]
    pos_samples = split_samples[[2]]
    loss = 0
    for(neg in neg_samples){
        for(pos in pos_samples){
            diff = neg+margin-pos
            if(diff>0)loss=loss+diff^2
        }
    }
    return(torch::torch_tensor(loss))
}

if(torch::torch_is_installed()){
test_that("All one class, unit test", {
    pred_tensor <- torch::torch_tensor(c(1,2,3,4,5,6,7))
    label_tensor <- torch::torch_tensor(c(1,1,1,1,1,1,1))
    margin <- 2
    expect_equal(mlr3torchAUM::LogLinearHSP(
        pred_tensor, label_tensor, margin),
        torch::torch_tensor(0))
    expect_equal(mlr3torchAUM::NaiveHSP(
        pred_tensor, label_tensor, margin),
        torch::torch_tensor(0))
})

test_that("All negative samples fall outside margin of all positive samples, unit test", {
    pred_tensor <- torch::torch_tensor(c(1,2,5,6,7,8))
    label_tensor <- torch::torch_tensor(c(0,0,1,1,1,1))
    margin <- 2
    ground_truth_loss = extremely_naive_HSP_loss(pred_tensor, label_tensor, margin)
    expect_equal(mlr3torchAUM::LogLinearHSP(
        pred_tensor, label_tensor, margin),
        ground_truth_loss)
    expect_equal(mlr3torchAUM::NaiveHSP(
        pred_tensor, label_tensor, margin),
        ground_truth_loss)
})

test_that("All negative samples fall inside margin of certain positive sample(s), unit test", {
    pred_tensor <- torch::torch_tensor(c(1,2,69,70,99,100))
    label_tensor <- torch::torch_tensor(c(0,1,0,1,0,1))
    margin <- 2
    ground_truth_loss = extremely_naive_HSP_loss(
        pred_tensor, label_tensor, margin)
    expect_equal(mlr3torchAUM::LogLinearHSP(
        pred_tensor, label_tensor, margin),
        ground_truth_loss)
    expect_equal(mlr3torchAUM::NaiveHSP(
        pred_tensor, label_tensor, margin),
        ground_truth_loss)
})

test_that("All negative samples are greater than all positive samples, unit test", {
    pred_tensor <- torch::torch_tensor(c(1,2,3,4,5,6,7))
    label_tensor <- torch::torch_tensor(c(1,1,1,1,0,0,0))
    margin <- 2
    ground_truth_loss = extremely_naive_HSP_loss(
        pred_tensor, label_tensor, margin)
    expect_equal(mlr3torchAUM::LogLinearHSP(
        pred_tensor, label_tensor, margin),
        ground_truth_loss)
    expect_equal(mlr3torchAUM::NaiveHSP(
        pred_tensor, label_tensor, margin),
        ground_truth_loss)
})

test_that("Some negative samples fall on margin of certain positive sample(s), unit test", {
    pred_tensor <- torch::torch_tensor(c(1,2,3,4,5,6,7))
    label_tensor <- torch::torch_tensor(c(1,0,1,0,1,1,1))
    margin <- 2
    ground_truth_loss = extremely_naive_HSP_loss(
        pred_tensor, label_tensor, margin)
    expect_equal(mlr3torchAUM::LogLinearHSP(
        pred_tensor, label_tensor, margin),
        ground_truth_loss)
    expect_equal(mlr3torchAUM::NaiveHSP(
        pred_tensor, label_tensor, margin),
        ground_truth_loss)
})
}
