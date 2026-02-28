## test for https://github.com/tdhock/mlr3torchAUM/issues/13
## Mock redirects torch_tensor() calls without device= onto "meta" device,
## reproducing the CUDA/CPU mismatch on CPU-only CI.

test_that("ROCAUM propagates input device to internal tensors", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())
  torch::torch_manual_seed(42L)
  pred <- torch::torch_randn(10)
  label_values <- sample(0:1, 10, replace = TRUE)
  while (length(unique(label_values)) < 2L) {
    label_values <- sample(0:1, 10, replace = TRUE)
  }
  label <- torch::torch_tensor(label_values)
  real_torch_tensor <- torch::torch_tensor
  local_mocked_bindings(
    torch_tensor = function(data, ..., device) {
      if (!missing(device)) return(real_torch_tensor(data, ..., device = device))
      real_torch_tensor(data, ..., device = "meta")
    },
    .package = "torch"
  )
  result <- ROCAUM(pred, label)
  expect_true(inherits(result, "torch_tensor"))
})
