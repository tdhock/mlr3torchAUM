# mlr3torchAUM

[[https://github.com/mlr3-imbalanced/mlr3torchAUM/actions/workflows/R-CMD-check.yaml][https://github.com/mlr3-imbalanced/mlr3torchAUM/actions/workflows/R-CMD-check.yaml/badge.svg?branch=maum-cran]]
[[https://app.codecov.io/gh/mlr3-imbalanced/mlr3torchAUM][https://codecov.io/gh/mlr3-imbalanced/mlr3torchAUM/branch/maum-cran/graph/badge.svg]]


Learning with Area Under the Minimum in the mlr3torch framework

## Installation

```r
remotes::install_github("tdhock/mlr3torchAUM")
```

## Usage

### New batch samplers

```r
mlr3torchAUM::batch_sampler_random(batch_size=9)
mlr3torchAUM::batch_sampler_stratified(min_samples_per_stratum=1)
```

Both can be used as the `batch_sampler` parameter in a `TorchLearner`, as below.

```r
L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
L$param_set$set_values(
  batch_sampler=mlr3torchAUM::batch_sampler_stratified(1))
```

Why do we need `batch_sampler_random`?
It uses torch randomness in a different way than the default sampler;
this method is compatible with `batch_sampler_stratified`,
so they can be used together for a controlled comparison between random and stratified sampling.

### New mlr3 measures

- Inverse AUC is 1-AUC, so we can visualize using a log scale and more easily see how close it gets to zero (=how close AUC gets to 1).
  This is a piecewise constant (non-differentiable) evaluation metric.
- ROC-AUM is Area Under Minimum of False Positive and False Negative Rates, see [our JMLR'23 paper](https://jmlr.org/papers/v24/21-0751.html) for details.
  This can be used as a surrogate loss for ROC curve optimization, because it is differentiable almost everywhere.
  This Measure is useful for monitoring how much it decreases in every epoch of learning, using the history callback.

```r
L$loss <- mlr3torchAUM::nn_ROCAUM_loss
L$param_set$set_values(
  measures_train=mlr3::msrs(c("classif.rocaum","classif.invauc")))
```

## Related work

- `mlr3torchAUM::batch_sampler_stratified` adapted from [this blog](https://tdhock.github.io/blog/2025/mlr3torch-batch-samplers/).
