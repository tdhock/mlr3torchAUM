register_mlr3 = function() {
  ## mlr_resamplings = utils::getFromNamespace("mlr_resamplings", ns = "mlr3")
  ## mlr_resamplings$add("same_other_sizes_cv", ResamplingSameOtherSizesCV)
  mlr3::mlr_measures$add("classif.rocaum", MeasureClassifROCAUM)
  mlr3::mlr_measures$add("classif.invauc", MeasureClassifInvAUC)
  mlr3::mlr_measures$add("classif.imcp", MeasureClassifIMCP)
  mlr3::mlr_measures$add("classif.mcp", MeasureClassifMCP)
}

register_mlr3torch = function(...) {
  mlr3torch::mlr3torch_losses$add("pairwise_auc_surrogate", torch_loss_pairwise_auc_surrogate)
  mlr3torch::mlr3torch_losses$add("sq_hinge_loglinear", torch_loss_sq_hinge_loglinear)
}

.onLoad = function(libname, pkgname) { # nolint
  # Configure Logger:
  assign("lg", lgr::get_logger("mlr3"), envir = parent.env(environment()))
  if (Sys.getenv("IN_PKGDOWN") == "true") {
    lg$set_threshold("warn") # nolint
  }
  x = utils::getFromNamespace("mlr_reflections", ns = "mlr3")
  x$loaded_packages = c(x$loaded_packages, "mlr3torchAUM")
  mlr3misc::register_namespace_callback(pkgname, "mlr3", register_mlr3)
  mlr3misc::register_namespace_callback(pkgname, "mlr3torch", register_mlr3torch)
}

# 'self' is injected by mlr3torch into callback stage functions (on_after_backward, ...)
utils::globalVariables("self")

mlr3misc::leanify_package()
