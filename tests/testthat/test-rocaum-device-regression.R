## test for https://github.com/tdhock/mlr3torchAUM/issues/13
## ROCAUM should work when inputs are on CUDA device.

test_that("ROCAUM propagates input device to internal tensors", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())
  skip_if_not(torch::cuda_is_available())
  
  pred <- torch::torch_randn(10)$cuda()
  label <- torch::torch_tensor(rep(0:1, 5))$cuda()
  
  result <- ROCAUM(pred, label)
  expect_is(result, "torch_tensor")
  expect_identical(result$device$type, "cuda")
})
