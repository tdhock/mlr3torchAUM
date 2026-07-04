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
            self$m_neighbors = m_neighbors
            self$out_step = out_step
        }
    ),
    private = list(
        .validate_estimator = function(){
            super$.validate_estimator()
            self$nn_m_ <- function(query, data) knn_index(query, data, self$m_neighbors)
        },
        .fit_resample = function(X, y) {}
    )
)