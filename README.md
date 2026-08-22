# mlr3torchAUM

[![R-CMD-check](https://github.com/mlr3-imbalanced/mlr3torchAUM/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/mlr3-imbalanced/mlr3torchAUM/actions/workflows/R-CMD-check.yaml)
[![codecov](https://codecov.io/gh/mlr3-imbalanced/mlr3torchAUM/graph/badge.svg)](https://app.codecov.io/gh/mlr3-imbalanced/mlr3torchAUM)


AUC/AUM-based losses, optimizers, batch samplers, over-samplers, and multiclass
`
performance measures for imbalanced classification in the mlr3torch framework.
## Installation

```r
remotes::install_github("mlr3-imbalanced/mlr3torchAUM")
```

## Usage

### New batch samplers

```r
mlr3torchAUM::batch_sampler_random(batch_size=9)
mlr3torchAUM::batch_sampler_stratified(min_samples_per_stratum=1)
mlr3torchAUM::batch_sampler_dual(pos_ratio=0.5, batch_size=32)
```

These can all be used as the `batch_sampler` parameter in a `TorchLearner`, as below.

```r
L <- mlr3torch::LearnerTorchMLP$new(task_type="classif")
L$param_set$set_values(
  batch_sampler=mlr3torchAUM::batch_sampler_stratified(1))
```

Why do we need `batch_sampler_random`?
It uses torch randomness in a different way than the default sampler;
this method is compatible with `batch_sampler_stratified`,
so they can be used together for a controlled comparison between random and stratified sampling.

`batch_sampler_dual` is a port of LibAUC's `DualSampler`: it guarantees a fixed
positive/negative ratio in every batch (oversampling the minority class once its
pool is exhausted), which is required by losses such as AUCM/CompositionalAUC
that need both classes present in each batch.

### New mlr3 measures

- Inverse AUC is 1-AUC, so we can visualize using a log scale and more easily see how close it gets to zero (=how close AUC gets to 1).
  This is a piecewise constant (non-differentiable) evaluation metric.
- ROC-AUM is Area Under Minimum of False Positive and False Negative Rates, see [our JMLR'23 paper](https://jmlr.org/papers/v24/21-0751.html) for details.
  This can be used as a surrogate loss for ROC curve optimization, because it is differentiable almost everywhere.
  This Measure is useful for monitoring how much it decreases in every epoch of learning, using the history callback.
- MeasureClassifIMCP provides (Imbalanced) Multiclass Classification Performance (MCP/IMCP) measures, generalizing AUM to the multiclass setting.

```r
L$loss <- mlr3torchAUM::nn_ROCAUM_loss
L$param_set$set_values(
  measures_train=mlr3::msrs(c("classif.rocaum","classif.invauc")))
```

### AUC-optimization losses ported from LibAUC

- `nn_AUCM_loss`: AUC-Margin loss, a surrogate for AUC maximization, trained with the
  `optim_pesg` (PESG) optimizer and the `make_pesg_callback` callback that updates its
  trainable parameters after every epoch.
- `nn_CompositionalAUC_loss`: alternates cross-entropy and AUC-margin batches, trained
  with the `optim_pdsca` (PDSCA) optimizer and the `make_pdsca_callback` callback.
- `nn_pairwise_auc_loss`: a pairwise surrogate loss for AUC optimization.
- `nn_squared_hinge_loss`: an all-pairs squared hinge log-linear loss, a convex
  surrogate for AUC computed in O(n log n) time via cumulative sums.

### Imbalanced classification: over-samplers

Ported from Python's [imbalanced-learn](https://imbalanced-learn.org/):

- `SMOTE`, `SMOTEN` (nominal/categorical features), `SMOTENC` (mixed nominal +
  continuous features), `SVMSMOTE` (SVM-support-vector-guided variant).
- `RandomOverSampler`, including the smoothed-bootstrap ROSE method.

### Multiclass classification performance: MCP/IMCP

- `nn_IMCP_loss` and `mcp_curve`: (Imbalanced) Multiclass Classification Performance
  losses and curves/areas, generalizing the AUM framework to multiclass problems.

## Related work

- Project blog: <https://mlr3torchaum-blog.netlify.app/>
- `mlr3torchAUM::batch_sampler_stratified` adapted from [this blog](https://tdhock.github.io/blog/2025/mlr3torch-batch-samplers/).
- `nn_AUCM_loss`, `nn_CompositionalAUC_loss`, `optim_pesg`, `optim_pdsca`, and
  `batch_sampler_dual` are R/torch ports of [LibAUC](https://github.com/Optimization-AI/LibAUC).
- `SMOTE`, `SMOTEN`, `SMOTENC`, `SVMSMOTE`, and `RandomOverSampler` are ports of
  [Python's imbalanced-learn](https://imbalanced-learn.org/).
- `nn_IMCP_loss` and `mcp_curve` are a torch port of [adaa-polsl/imcp](https://github.com/adaa-polsl/imcp).

## Contributions by @weicaocw

The following features were implemented by [@weicaocw](https://github.com/weicaocw):

- **AUC-margin loss and PESG/PDSCA optimization** (PR #55, #59, #68, #90, #93): `nn_AUCM_loss`
  (with `v1`/`v2` variants and the `imratio` class-prior argument), `optim_pesg` and
  `make_pesg_callback`; `nn_CompositionalAUC_loss` with `optim_pdsca` and
  `make_pdsca_callback`; `batch_sampler_dual`.
- **(Imbalanced) Multiclass Classification Performance, MCP/IMCP** (PR #50, #52, #111):
  migrated `adaa-polsl/imcp` into a torch version — `nn_IMCP_loss`, `mcp_curve`, and
  the `MeasureClassifIMCP` mlr3 measures, plus documentation.
- **SMOTE family of over-samplers** (PR #69, #80, #82, #83, #84): `SMOTE`, `SVMSMOTE`,
  `SMOTEN`, `SMOTENC`.
- **Package infrastructure** (PR #37, #38, #39, #42): documentation website
  (litedown + Netlify), CI cache fixes, CRAN-compliance cleanup, and README/branch tidy-up
  in the test coverage workflow.
