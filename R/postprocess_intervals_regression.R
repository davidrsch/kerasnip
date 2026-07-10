# Laplace interval prediction functions and post-processing (regression)
#
# laplace_conf_int_reg() and
# laplace_pred_int_reg() are exported (@keywords internal) so
# that parsnip can call them via c(pkg = "kerasnip", fun = "...").
#
# postprocess_intervals_reg() formats the raw interval matrices
# into a parsnip-standard tibble.

#' Epistemic Variance from Penultimate Features (Keras Ops)
#'
#' @description
#' Computes per-sample epistemic (weight-uncertainty-only) variance from
#' already-computed penultimate-layer features, using Keras ops. Split out
#' from `laplace_step_moments()` so callers that already have `features`
#' from a shared forward pass (e.g. `laplace_joint_pred_int()`, where every
#' step reads the same `combined_model`) don't need to re-run `predict()`
#' once per step.
#'
#' @param features An (n, d) numeric matrix of penultimate-layer features.
#' @param h_diag Numeric vector of diagonal GGN entries.
#' @param tau Scalar prior precision.
#' @param sigma_sq_noise Scalar observation noise variance.
#' @param n_training Integer, number of training points.
#' @return A numeric vector of epistemic variances, one per row of `features`.
#' @noRd
laplace_epistemic_variance <- function(
  features,
  h_diag,
  tau,
  sigma_sq_noise,
  n_training
) {
  d_features <- ncol(features)
  h_w <- h_diag[1L:d_features]
  h_b <- h_diag[d_features + 1L]

  features_k <- keras3::op_convert_to_tensor(features, dtype = "float32")
  prec_w <- n_training * h_w / sigma_sq_noise + tau
  prec_b <- n_training * h_b / sigma_sq_noise + tau

  var_k <- keras3::op_matmul(
    keras3::op_square(features_k),
    keras3::op_convert_to_tensor(1 / prec_w, dtype = "float32")
  )
  as.array(var_k) + 1 / prec_b
}

#' Per-Sample Mean and Epistemic Variance for a Single Regression Output
#'
#' @description
#' Given a combined model (returning both predictions and features in one
#' forward pass), computes the per-sample point prediction and epistemic
#' variance (via `laplace_epistemic_variance()`). Used by
#' `build_intervals_regression()` for marginal conf_int/pred_int.
#'
#' @inheritParams build_intervals_regression
#' @return A list with `mean` (numeric vector) and `var_epistemic` (numeric
#'   vector), one entry per row of `x`.
#' @noRd
laplace_step_moments <- function(
  combined_model,
  x,
  h_diag,
  tau,
  sigma_sq_noise,
  n_training,
  col_idx = NULL
) {
  combined_pred <- predict(combined_model, x)
  mean_pred <- if (is.null(col_idx)) {
    as.vector(combined_pred$pred)
  } else {
    as.matrix(combined_pred$pred)[, col_idx]
  }
  features <- as.matrix(combined_pred$features)
  var_vec <- laplace_epistemic_variance(
    features,
    h_diag,
    tau,
    sigma_sq_noise,
    n_training
  )

  list(mean = mean_pred, var_epistemic = var_vec)
}


#' Build Intervals for a Single Regression Output
#'
#' @description
#' Computes per-sample epistemic or predictive variance (via
#' `laplace_step_moments()`) and returns symmetric Normal-based intervals.
#'
#' @param combined_model A Keras model with outputs `pred` and `features`.
#' @param x Processed predictor data (matrix or array).
#' @param h_diag Numeric vector of diagonal GGN entries.
#' @param tau Scalar prior precision.
#' @param sigma_sq_noise Scalar observation noise variance.
#' @param n_training Integer, number of training points.
#' @param level Confidence level (e.g. 0.95).
#' @param predictive Logical; if TRUE adds sigma_sq_noise (pred_int).
#' @param col_idx Integer, optional. For a vector-valued (multi-step)
#'   output, the column of `combined_model`'s `pred` output this entry's
#'   posterior belongs to. `NULL` (default) for a scalar output, where
#'   `pred` already has a single column.
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
  predictive = FALSE,
  col_idx = NULL
) {
  moments <- laplace_step_moments(
    combined_model,
    x,
    h_diag,
    tau,
    sigma_sq_noise,
    n_training,
    col_idx
  )
  mean_pred <- moments$mean
  var_vec <- moments$var_epistemic

  if (predictive) {
    var_vec <- var_vec + sigma_sq_noise
  }

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
      predictive = FALSE,
      col_idx = entry$col_idx
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
      predictive = TRUE,
      col_idx = entry$col_idx
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
#' For a single vector-valued (multi-step) output, `results` is instead a
#' named list keyed `step_1`, `step_2`, ... (see
#' `laplace_multistep_regression()`); these are nested into the same
#' `.pred` list-column convention `keras_postprocess_numeric()` uses for
#' multi-step point predictions, one inner tibble per row with `.step` plus
#' `.pred`/`.pred_lower`/`.pred_upper` (or their `_<var>` suffixed forms for
#' multiple forecasted variables).
#'
#' @param results A matrix (single-output) or a named list of matrices
#'   (multi-output, or multi-step; see `object$fit$multistep_info`).
#' @param object The `model_fit` object; `object$fit$multistep_info`
#'   distinguishes the multi-step case from genuine multi-output.
#' @return A tibble.
#' @noRd
postprocess_intervals_reg <- function(results, object) {
  if (
    is.list(results) &&
      !is.null(names(results)) &&
      !inherits(results, "data.frame")
  ) {
    if (!is.null(object$fit$multistep_info)) {
      return(multistep_interval_pred_column(results, object$fit$multistep_info))
    }
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

#' Build a Nested `.pred` Column for Vector-Valued (Multi-Step) Intervals
#'
#' @description
#' Given `results`, a named list `step_1`, `step_2`, ... of interval
#' matrices (each with columns `.pred`, `.pred_lower`, `.pred_upper`, one
#' row per sample), builds a `.pred` list-column with one inner tibble per
#' row: a `.step` column plus the interval columns per forecasted variable
#' (unsuffixed if there is only one, `_<var>` suffixed if there are
#' several), mirroring `multistep_pred_column()` in
#' `register_fit_predict.R` for point predictions.
#'
#' @param results A named list of interval matrices, one per step (in
#'   the same column order as the original outcome).
#' @param multistep_info A list with `steps` and `vars`, as produced by
#'   `parse_multistep_column_names()`.
#' @return A tibble with one `.pred` list-column.
#' @noRd
multistep_interval_pred_column <- function(results, multistep_info) {
  steps <- multistep_info$steps
  vars <- multistep_info$vars
  uniq_steps <- sort(unique(steps))
  uniq_vars <- unique(vars)
  n_rows <- nrow(results[[1L]])
  interval_cols <- c(".pred", ".pred_lower", ".pred_upper")

  pred_list <- lapply(seq_len(n_rows), function(i) {
    row_df <- tibble::tibble(.step = uniq_steps)
    for (v in uniq_vars) {
      suffix <- if (length(uniq_vars) > 1) paste0("_", v) else ""
      idx <- multistep_var_col_order(vars, steps, v)
      for (col in interval_cols) {
        row_df[[paste0(col, suffix)]] <- vapply(
          idx,
          function(k) results[[k]][i, col],
          numeric(1)
        )
      }
    }
    row_df
  })

  tibble::tibble(.pred = pred_list)
}

#' Draw Correlated Noise Samples from a Covariance Matrix
#'
#' @description
#' Draws `n_draws` samples from `MVN(0, sigma)` via a Cholesky factor, so
#' that (for `k > 1`) the columns of the result are correlated according to
#' `sigma`'s off-diagonal entries. Separated from `laplace_joint_pred_int()`
#' so the sampling math itself can be unit-tested against a known,
#' synthetic covariance matrix, independent of any fitted Keras model.
#'
#' @param n_draws Integer, number of samples to draw.
#' @param sigma A `(k, k)` numeric covariance matrix.
#' @return An `(n_draws, k)` numeric matrix, each row a draw from
#'   `MVN(0, sigma)`.
#' @noRd
sample_correlated_noise <- function(n_draws, sigma) {
  k <- ncol(sigma)
  # chol() returns the upper-triangular U with t(U) %*% U == sigma, so
  # z %*% U (z's rows ~ MVN(0, I)) gives rows ~ MVN(0, sigma).
  u <- chol(sigma)
  z <- matrix(stats::rnorm(n_draws * k), nrow = n_draws, ncol = k)
  z %*% u
}

#' Joint (Correlated) Multi-Step Prediction Intervals via Sampling
#'
#' @description
#' For a vector-valued (multi-step) regression output, draws correlated
#' sample trajectories across all forecast steps: each draw combines every
#' step's own independent epistemic variance (via
#' `laplace_epistemic_variance()`, from its last-layer Laplace posterior)
#' with a *jointly* sampled aleatoric noise vector from the empirical
#' cross-step residual covariance computed at fit time (`joint_noise_cov`,
#' see `laplace_multistep_regression()`). Only `type = "pred_int"` is
#' supported: `type = "conf_int"` reflects epistemic uncertainty only, and
#' this implementation has no estimated source of cross-step correlation
#' for that case.
#'
#' @details
#' Returns raw draws (one row per sample, per step, per draw) rather than a
#' pre-summarized interval, matching the `tidybayes`/tidyverse convention
#' (`.chain`/`.iteration`/`.draw`) for "several posterior/predictive samples
#' per observation", rather than inventing a new column convention. Users
#' derive whatever joint or marginal summary they need (e.g. joint
#' quantile bands, or `group_by(.step) |> summarize(...)` to recover
#' marginal intervals) with standard tidyverse tools.
#'
#' @param object A `kerasnip_model_fit` object.
#' @param new_data A data frame of predictors.
#' @param n_draws Integer, number of joint sample trajectories per row
#'   (default 1000).
#' @return A tibble with one `.pred` list-column; each element is a tibble
#'   with `.draw`, `.step`, and `.pred` (long format, `n_draws * n_steps`
#'   rows).
#' @noRd
laplace_joint_pred_int <- function(object, new_data, n_draws = 1000L) {
  laplace_data <- object$fit$laplace
  if (is.null(laplace_data)) {
    rlang::abort(c(
      "Laplace prediction intervals are not available for this model.",
      i = "The model was not fitted with a Dense output layer,",
      i = "or the Laplace posterior could not be computed.",
      i = "Re-fit the model to enable interval predictions."
    ))
  }
  sigma_noise_joint <- attr(laplace_data, "joint_noise_cov")
  if (is.null(sigma_noise_joint)) {
    rlang::abort(c(
      "`joint = TRUE` is only supported for vector-valued (multi-step) ",
      "regression outputs.",
      i = "Use the default `joint = FALSE` for this model."
    ))
  }

  x <- object$fit$process_x(new_data)$x_proc
  n_steps <- length(laplace_data)

  # Every step's `combined_model` is the same shared object built once in
  # laplace_multistep_regression() (same underlying Keras layers, so the
  # same forward pass gives every step's prediction and the shared
  # penultimate features in one call). Calling predict() once here rather
  # than once per step avoids n_steps redundant Keras forward passes on
  # identical input.
  combined_pred <- predict(laplace_data[[1L]]$combined_model, x)
  means <- as.matrix(combined_pred$pred)
  features <- as.matrix(combined_pred$features)
  n_obs <- nrow(means)

  vars_epistemic <- matrix(NA_real_, nrow = n_obs, ncol = n_steps)
  for (k in seq_len(n_steps)) {
    entry <- laplace_data[[k]]
    vars_epistemic[, k] <- laplace_epistemic_variance(
      features,
      entry$h_diag,
      entry$tau,
      entry$sigma_sq_noise,
      entry$n_training
    )
  }

  pred_list <- lapply(seq_len(n_obs), function(i) {
    # Each observation draws its own independent noise sample; sharing one
    # draw across observations would artificially correlate different
    # rows' simulated forecast errors.
    noise_draws <- sample_correlated_noise(n_draws, sigma_noise_joint)
    epistemic_draws <- matrix(
      stats::rnorm(n_draws * n_steps),
      nrow = n_draws,
      ncol = n_steps
    ) *
      matrix(
        sqrt(pmax(vars_epistemic[i, ], 0)),
        nrow = n_draws,
        ncol = n_steps,
        byrow = TRUE
      )
    draws_mat <- matrix(
      means[i, ],
      nrow = n_draws,
      ncol = n_steps,
      byrow = TRUE
    ) +
      epistemic_draws +
      noise_draws

    tibble::tibble(
      .draw = rep(seq_len(n_draws), times = n_steps),
      .step = rep(seq_len(n_steps), each = n_draws),
      .pred = as.vector(draws_mat)
    )
  })

  tibble::tibble(.pred = pred_list)
}
