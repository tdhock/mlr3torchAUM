# benchmark/scripts/R/learners.R

LearnerTorchMLPTrainSampler <- R6::R6Class(
  "LearnerTorchMLPTrainSampler",
  inherit = mlr3torch::LearnerTorchMLP,
  private = list(
    .dataloader_predict = function(dataset, param_vals) {
      param_vals$batch_sampler <- NULL
      param_vals$sampler <- NULL
      super$.dataloader_predict(dataset, param_vals)
    }
  )
)

LearnerTorchModuleTrainSampler <- R6::R6Class(
  "LearnerTorchModuleTrainSampler",
  inherit = mlr3torch::LearnerTorchModule,
  private = list(
    .dataloader_predict = function(dataset, param_vals) {
      param_vals$batch_sampler <- NULL
      param_vals$sampler <- NULL
      super$.dataloader_predict(dataset, param_vals)
    }
  )
)

LearnerTorchModuleNoStratum <- R6::R6Class(
  "LearnerTorchModuleNoStratum",
  inherit = mlr3torchAUM::LearnerTorchModuleTrainSampler,
  public = list(
    train = function(task, row_ids = NULL) {
      if (length(task$col_roles$group) && length(task$col_roles$stratum)) {
        task <- task$clone(deep = TRUE)
        task$col_roles$stratum <- character(0)
      }
      super$train(task, row_ids)
    }
  )
)

AutoTunerTorchX <- R6::R6Class(
  "AutoTunerTorchX",
  inherit = mlr3tuning::AutoTuner,
  public = list(
    module_learner = NULL,
    initialize = function(id, max_epochs, measure_list, validate = 0.5,
                          extra_callbacks = list(), configure = NULL, ...) {
      if (!is.list(measure_list)) measure_list <- list(measure_list)
      M <- measure_list[[1L]]
      task_type <- sub("[.].*", "", M$id)
      ## put the "history" callback first
      ## append the additional callbacks next, forming a list
      cb_list <- c(list(mlr3torch::t_clbk("history")), extra_callbacks)
      ## Should differentiate the parameters in "..."...`
      ## One part is for the $new: module_generator / loss / optimizer / …）
      ## Another for param_set: batch_size / batch_sampler / sampler / …）
      dots <- list(...)
      ctor_names <- names(formals(mlr3torch::LearnerTorchModule$public_methods$initialize))
      is_ctor <- names(dots) %in% ctor_names
      ctor_args <- dots[is_ctor]
      ps_args <- dots[!is_ctor]
      ps_args <- ps_args[!vapply(ps_args, is.null, logical(1))]
      # We don't use mlr3::lrn("classif.module")
      # but customized LearnerTorchModuleNoStratum
      self$module_learner <- do.call(
        LearnerTorchModuleNoStratum$new,
        c(
          list(
            task_type = task_type,
            ingress_tokens = list(x = mlr3torch::ingress_num()),
            callbacks = cb_list
          ),
          ctor_args
        )
      )
      do.call(self$module_learner$param_set$set_values, c(list(
        epochs = paradox::to_tune(upper = max_epochs, internal = TRUE),
        patience = max_epochs,
        measures_valid = measure_list,
        measures_train = measure_list
      ), ps_args))
      if (task_type == "classif") self$module_learner$predict_type <- "prob"
      # Cannot tolerate group and stratum
      mlr3::set_validate(self$module_learner, validate = validate)
      ## Should complete configuring the module learner
      ## BEFORE super$initialize()
      if (is.function(configure)) configure(self$module_learner)
      terminator <- mlr3tuning::mlr_terminators$get("evals")
      terminator$param_set$set_values(n_evals = 1)
      super$initialize(
        learner = self$module_learner,
        tuner = mlr3tuning::tnr("internal"),
        resampling = mlr3::rsmp("insample"), # ATTENTION! another resample here
        measure = mlr3::msr("internal_valid_score", minimize = TRUE),
        terminator = terminator,
        id = id,
        store_models = TRUE
      )
    },
    save_learner = function() {
      list(history = self$archive$learners(1)[[1]]$model$callbacks$history)
    },
    # See "ATTENTION! another resample here"
    # The row_ids is distributed with proj_compute using SOAK
    # So it is safe to take it off here
    train = function(task, row_ids = NULL) {
      if (length(task$col_roles$group) && length(task$col_roles$stratum)) {
        task <- task$clone(deep = TRUE)
        task$col_roles$stratum <- character(0)
      }
      super$train(task, row_ids)
    },
    edit_learner = function() {
      self$learner$param_set$set_values(
        patience = 2L,
        epochs = paradox::to_tune(upper = 2L, internal = TRUE)
      )
    }
  )
)

linear_module <- function() {
  torch::nn_module(
    "linear",
    initialize = function(task) {
      self$fc <- torch::nn_linear(length(task$feature_names), 1L)
    },
    forward = function(x) self$fc(x)
  )
}

BATCH <- 500L
LR <- 0.1

sgd_tuner <- function(id, loss, max_epochs, measure_list, ...) {
  AutoTunerTorchX$new(
    id = id, max_epochs = max_epochs, measure_list = measure_list,
    module_generator = linear_module(),
    loss = loss,
    optimizer = mlr3torch::t_opt("sgd", lr = LR),
    batch_size = BATCH,
    ...
  )
}

aucm_tuner <- function(id, version, max_epochs, measure_list, dual = FALSE, imratio = NULL) {
  AutoTunerTorchX$new(
    id = id, max_epochs = max_epochs, measure_list = measure_list,
    extra_callbacks = list(mlr3torchAUM::make_pesg_callback_full()),
    configure = function(ml) {
      mlr3torchAUM::pesg_config(
        ml,
        lr = 0.05, clamp_value = 1, weight_decay = 1e-5,
        epoch_decay = 2e-3, momentum = 0.9, decay_factor = 1
      )
    },
    module_generator = linear_module(),
    loss = mlr3torch::t_loss("aucm", version = version, add_sigmoid = TRUE, imratio = imratio),
    optimizer = mlr3torch::as_torch_optimizer(mlr3torchAUM::optim_pesg),
    batch_size = BATCH,
    batch_sampler = if (dual) {
      mlr3torchAUM::batch_sampler_dual(batch_size = BATCH, sampling_rate = 0.5)
    } else {
      NULL
    }
  )
}

compo_tuner <- function(id, version, max_epochs, measure_list, dual = FALSE, imratio = NULL) {
  AutoTunerTorchX$new(
    id = id, max_epochs = max_epochs, measure_list = measure_list,
    extra_callbacks = list(mlr3torchAUM::make_pdsca_callback_full()),
    configure = function(ml) mlr3torchAUM::pdsca_config(ml),
    module_generator = linear_module(),
    loss = mlr3torch::t_loss("compositional_auc", version = version, add_sigmoid = TRUE, imratio = imratio),
    optimizer = mlr3torch::as_torch_optimizer(mlr3torchAUM::optim_pdsca),
    batch_size = BATCH,
    batch_sampler = if (dual) {
      mlr3torchAUM::batch_sampler_dual(batch_size = BATCH, sampling_rate = 0.5)
    } else {
      NULL
    }
  )
}

learner_list <- function(max_epochs = 200L,
                         measure_list = mlr3::msrs(c("classif.auc", "classif.acc")),
                         only = NULL,
                         imratio = NULL) {
  L <- list()
  L[["linear_ce"]] <- sgd_tuner(
    "linear_ce", mlr3torch::t_loss("cross_entropy"), max_epochs, measure_list
  )
  L[["linear_AUM"]] <- sgd_tuner(
    "linear_AUM", mlr3torch::t_loss("rocaum"), max_epochs, measure_list
  )
  L[["linear_AUCM_v1"]] <- aucm_tuner("linear_AUCM_v1", "v1", max_epochs, measure_list, dual = FALSE, imratio = imratio)
  L[["linear_AUCM_v2"]] <- aucm_tuner("linear_AUCM_v2", "v2", max_epochs, measure_list, dual = TRUE, imratio = imratio)
  L[["linear_Comp_v1"]] <- aucm_tuner("linear_Comp_v1", "v1", max_epochs, measure_list, dual = FALSE, imratio = imratio)
  L[["linear_Comp_v2"]] <- aucm_tuner("linear_Comp_v2", "v2", max_epochs, measure_list, dual = TRUE, imratio = imratio)
  L[["linear_pair_sqhinge"]] <- sgd_tuner(
    "linear_pair_sqhinge",
    mlr3torch::t_loss("pairwise_auc_surrogate", surr_loss = "squared_hinge"),
    max_epochs, measure_list
  )
  L[["linear_pair_logistic"]] <- sgd_tuner(
    "linear_pair_logistic",
    mlr3torch::t_loss("pairwise_auc_surrogate", surr_loss = "logistic"),
    max_epochs, measure_list
  )
  L[["linear_sqh_loglin"]] <- sgd_tuner(
    "linear_sqh_loglin",
    mlr3torch::t_loss("sq_hinge_loglinear"),
    max_epochs, measure_list
  )
  ## Those non-torch baselines' save_learner() will write out learners_weights.csv
  L[["cv_glmnet"]] <- mlr3resampling::LearnerClassifCVGlmnetSave$new()
  fl <- mlr3::LearnerClassifFeatureless$new()
  fl$predict_type <- "prob"
  L[["featureless"]] <- fl
  if (!is.null(only)) {
    miss <- setdiff(only, names(L))
    if (length(miss)) stop("Unknown learner+loss: ", paste(miss, collapse = ", "))
    L <- L[only]
  }
  L
}
