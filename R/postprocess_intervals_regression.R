# Laplace interval prediction functions and post-processing (regression)
#
# laplace_conf_int_reg() and
# laplace_pred_int_reg() are exported (@keywords internal) so
# that parsnip can call them via c(pkg = "kerasnip", fun = "...").
#
# postprocess_intervals_reg() formats the raw interval matrices
# into a parsnip-standard tibble.

#' Build Intervals for a Single Regression Output
#'
#' @description
#' Given a combined model (returning both predictions and features in one
#' forward pass), computes per-sample epistemic or predictive variance and
#' returns symmetric Normal-based intervals.
#'
#' @param combined_model A Keras model with outputs `pred` and `features`.
#' @param x Processed predictor data (matrix or array).
#' @param h_diag Numeric vector of diagonal GGN entries.
#' @param tau Scalar prior precision.
#' @param sigma_sq_noise Scalar observation noise variance.
#' @param n_training Integer, number of training points.
#' @param level Confidence level (e.g. 0.95).
#' @param variance_fn Function to compute per-sample variance
#'   (`epistemic_var_regression` or
#'   `predictive_var_regression`).
#' @return A matrix with columns `.pred`, `.pred_lower`, `.pred_upper`.
#' @noRd
build_intervals_regression <- function(
  combined_model,
  x,
  h_diag,
  tau,
  sigma_sq_noise,
  n_training,
  level,
  variance_fn
) {
  combined_pred <- predict(combined_model, x)

  if (is.list(combined_pred) && !is.null(names(combined_pred))) {
    mean_pred <- as.vector(combined_pred$pred)
    features <- as.matrix(combined_pred$features)
  } else {
    mean_pred <- as.vector(combined_pred)
    features <- as.matrix(combined_pred)
  }

  var_vec <- variance_fn(features, h_diag, tau, sigma_sq_noise, n_training)
  std_err <- sqrt(pmax(var_vec, 0))

  z <- stats::qnorm(1 - (1 - level) / 2)

  cbind(
    .pred = mean_pred,
    .pred_lower = mean_pred - z * std_err,
    .pred_upper = mean_pred + z * std_err
  )
}


#' Predict Confidence Intervals via Last-Layer Laplace Approximation
#'
#' @description
#' Predict-time entry point for regression confidence intervals.  Called by
#' parsnip via
#' `c(pkg = "kerasnip", fun = "laplace_conf_int_reg")`.
#'
#' For each output in the model, this builds the per-sample epistemic variance
#' (uncertainty on `E[Y|X]`) from the stored Laplace posterior and returns
#' symmetric Normal-based intervals at the requested confidence level.
#'
#' @param object The raw Keras model (from `object$fit$fit`).
#' @param x Processed predictor data (matrix or array).
#' @param laplace_data A named list of Laplace posterior data, one entry per
#'   output (from `object$fit$laplace`).  Each entry contains `h_diag`, `tau`,
#'   `sigma_sq_noise`, `n_training`, and `combined_model`.
#' @param level Confidence level (default 0.95).  Passed through from
#'   `predict(..., type = "conf_int", level = 0.95)`.
#' @return A named list of matrices (one per output), each with columns
#'   `.pred`, `.pred_lower`, and `.pred_upper`.
#' @export
#' @keywords internal
laplace_conf_int_reg <- function(
  object,
  x,
  laplace_data,
  level = 0.95
) {
  if (is.null(laplace_data)) {
    rlang::abort(c(
      "Laplace confidence intervals are not available for this model.",
      i = "The model was not fitted with a Dense output layer,",
      i = "or the Laplace posterior could not be computed.",
      i = "Re-fit the model to enable interval predictions."
    ))
  }

  lapply(laplace_data, function(entry) {
    build_intervals_regression(
      combined_model = entry$combined_model,
      x = x,
      h_diag = entry$h_diag,
      tau = entry$tau,
      sigma_sq_noise = entry$sigma_sq_noise,
      n_training = entry$n_training,
      level = level,
      variance_fn = epistemic_var_regression
    )
  })
}


#' Predict Prediction Intervals via Last-Layer Laplace Approximation
#'
#' @description
#' Predict-time entry point for regression prediction intervals.  Called by
#' parsnip via
#' `c(pkg = "kerasnip", fun = "laplace_pred_int_reg")`.
#'
#' For each output in the model, this builds the per-sample predictive variance
#' (uncertainty on a new observation `Y|X` = epistemic variance + observation
#' noise) from the stored Laplace posterior and returns symmetric Normal-based
#' intervals at the requested level.
#'
#' @inheritParams laplace_conf_int_reg
#' @return A named list of matrices (one per output), each with columns
#'   `.pred`, `.pred_lower`, and `.pred_upper`.
#' @export
#' @keywords internal
laplace_pred_int_reg <- function(
  object,
  x,
  laplace_data,
  level = 0.95
) {
  if (is.null(laplace_data)) {
    rlang::abort(c(
      "Laplace prediction intervals are not available for this model.",
      i = "The model was not fitted with a Dense output layer,",
      i = "or the Laplace posterior could not be computed.",
      i = "Re-fit the model to enable interval predictions."
    ))
  }

  lapply(laplace_data, function(entry) {
    build_intervals_regression(
      combined_model = entry$combined_model,
      x = x,
      h_diag = entry$h_diag,
      tau = entry$tau,
      sigma_sq_noise = entry$sigma_sq_noise,
      n_training = entry$n_training,
      level = level,
      variance_fn = predictive_var_regression
    )
  })
}


#' Post-Process Interval Matrices into a Parsnip-Standard Tibble
#'
#' @description
#' Formats raw interval prediction results into a tibble.
#'
#' @details
#' For single-output models `results` is a matrix with columns `.pred`,
#' `.pred_lower`, `.pred_upper` and is converted directly to a tibble.
#'
#' For multi-output models `results` is a named list of such matrices; they
#' are combined into a single tibble with per-output column names:
#' `.pred_<name>`, `.pred_lower_<name>`, `.pred_upper_<name>`.
#'
#' @param results A matrix (single-output) or a named list of matrices
#'   (multi-output).
#' @param object The `model_fit` object (unused; required by parsnip
#'   convention).
#' @return A tibble.
#' @noRd
postprocess_intervals_reg <- function(results, object) {
  if (is.list(results) && !is.null(names(results))) {
    is_multi <- length(results) > 1L
    combined_preds <- lapply(names(results), function(nm) {
      mat <- results[[nm]]
      if (is_multi) {
        colnames(mat) <- paste0(
          c(".pred", ".pred_lower", ".pred_upper"),
          "_",
          nm
        )
      }
      tibble::as_tibble(mat)
    })
    do.call(tibble::tibble, combined_preds)
  } else {
    if (is.list(results)) {
      mat <- results[[1L]]
    } else {
      mat <- results
    }
    tibble::as_tibble(mat)
  }
}
