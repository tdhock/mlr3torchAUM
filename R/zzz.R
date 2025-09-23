register_mlr3 = function() {
  ## mlr_resamplings = utils::getFromNamespace("mlr_resamplings", ns = "mlr3")
  ## mlr_resamplings$add("same_other_sizes_cv", ResamplingSameOtherSizesCV)
  mlr3::mlr_measures$add("classif.rocaum", MeasureClassifROCAUM)
  mlr3::mlr_measures$add("classif.invauc", MeasureClassifInvAUC)
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
}

mlr3misc::leanify_package()
