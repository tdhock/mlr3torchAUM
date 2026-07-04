minority_support_vectors <- function(svm_index, y, target_class) {
  return(svm_index[as.character(y[svm_index]) == target_class])
}

fit_svm <- function(X, y) {
  if (!requireNamespace("e1071")) stop("e1071 not installed ")
  gamma <- 1 / (ncol(X) * (mean(X^2) - mean(X)^2))
  e1071::svm(x = X, y = y, kernel = "radial", cost = 1, gamma = gamma, scale = FALSE)
}

SVMSMOTE <- R6::R6Class(
  "SVMSMOTE",
  inherit = BaseSMOTE,
  public = list(
    m_neighbors = NULL,
    out_step = NULL,
    nn_m_ = NULL,
    initialize = function(sampling_strategy = "auto", k_neighbors = 5,
                          m_neighbors = 10, out_step = 0.5) {
      super$initialize(sampling_strategy, k_neighbors)
      self$m_neighbors <- m_neighbors
      self$out_step <- out_step
    }
  ),
  private = list(
    .validate_estimator = function() {
      super$.validate_estimator()
      if (!is.numeric(self$m_neighbors) || length(self$m_neighbors) != 1 ||
        self$m_neighbors < 1 || self$m_neighbors != round(self$m_neighbors)) {
        stop("m_neighbors must be a positive integer")
      }
      self$nn_m_ <- function(query, data) knn_index(query, data, self$m_neighbors)
    },
    .fit_resample = function(X, y) {
      private$.validate_estimator()
      svm_index <- fit_svm(X, y)$index
      results <- lapply(names(self$sampling_strategy_), function(class_name) {
        minority_svs <- minority_support_vectors(svm_index, y, class_name)
        minority_svs_nn_idx <- self$nn_m_(X[minority_svs, , drop = FALSE], X)
        noise_sv_mask <- in_danger_noise(minority_svs_nn_idx, y, class_name, kind = "noise")
        minority_svs <- minority_svs[!noise_sv_mask] # exclude noise
        minority_svs_nn_idx <- minority_svs_nn_idx[!noise_sv_mask, , drop = FALSE]
        danger_sv_mask <- in_danger_noise(minority_svs_nn_idx, y, class_name, kind = "danger")
        danger_svs <- minority_svs[danger_sv_mask]
        safe_svs <- minority_svs[!danger_sv_mask]
        X_within_class <- X[as.character(y) == class_name, , drop = FALSE]
        n_to_generate <- self$sampling_strategy_[[class_name]]
        if (n_to_generate <= 0) next # won't be here
        n_danger <- length(danger_svs)
        n_safe <- length(safe_svs)
        if (n_danger == 0L && n_safe == 0L) {
          warning(sprintf(
            "SVMSMOTE: all support vectors of class '%s' are noise; generated 0 samples for it.",
            class_name
          ))
          return(list(X = X[integer(0), , drop = FALSE], y = character(0)))
        }
        if (n_danger == 0L) {
          message(sprintf(
            "SVMSMOTE: class '%s' has no danger support vectors; all %d generated samples come from the safe ones (extrapolation).",
            class_name, n_to_generate
          ))
          n_gen_danger <- 0L
          n_gen_safe <- n_to_generate
        } else if (n_safe == 0L) {
          message(sprintf(
            "SVMSMOTE: class '%s' has no safe support vectors; all %d generated samples come from the danger ones (interpolation).",
            class_name, n_to_generate
          ))
          n_gen_danger <- n_to_generate
          n_gen_safe <- 0L
        } else {
          frac <- stats::rbeta(1, 10, 10)
          n_gen_danger <- as.integer(frac * (n_to_generate + 1))
          n_gen_safe <- n_to_generate - n_gen_danger
        }
        generated_danger <- X[integer(0), , drop = FALSE]
        generated_safe <- X[integer(0), , drop = FALSE]
        if (n_gen_danger > 0L) {
          generated_danger <- private$.make_samples(
            X[danger_svs, , drop = FALSE],
            self$nn_k_(query = X[danger_svs, , drop = FALSE], data = X_within_class),
            n_gen_danger,
            nn_data = X_within_class
          )
        }
        if (n_gen_safe > 0L) {
          generated_safe <- private$.make_samples(
            X[safe_svs, , drop = FALSE],
            self$nn_k_(query = X[safe_svs, , drop = FALSE], data = X_within_class),
            n_gen_safe,
            nn_data = X_within_class, step_size = -self$out_step
          )
        }
        generated <- rbind(generated_danger, generated_safe)
        return(list(X = generated, y = rep(class_name, nrow(generated))))
      })
      return(list(
        X = do.call(rbind, c(list(X), lapply(results, function(result) result$X))),
        y = factor(c(as.character(y), unlist(lapply(results, function(result) result$y))),
          levels = levels(y)
        )
      ))
    }
  )
)
