# Laplace interval prediction functions for classification
#
# laplace_conf_int_cls() and
# laplace_pred_int_cls() are exported (@keywords internal)
# so that parsnip can call them via c(pkg = "kerasnip", fun = "...").
#
# Interval construction uses Monte Carlo sampling from the Laplace posterior
# over logits, followed by sigmoid / softmax transformation (and, for
# pred_int, Bernoulli / Categorical sampling), then per-class quantile
# computation.
#
# The column naming follows the parsnip convention:
#   .pred_lower_LevelA, .pred_upper_LevelA,
#   .pred_lower_LevelB, .pred_upper_LevelB, ...

#' Per-Class Logit Variance from Diagonal Laplace Posterior
#'
#' @description
#' Computes per-sample, per-class logit variance from the stored diagonal
#' GGN entries and prior precision `tau`.
#'
#' @details
#' The `h_diag` vector is laid out as `[w_1c, ..., w_dc, b_c]` repeated for
#' each of `C` classes, giving a total length of `C * (d + 1)`.
#'
#' @param features An (M x d) numeric matrix of penultimate-layer features.
#' @param h_diag Numeric vector of length `C * (d + 1)`.  Diagonal GGN
#'   entries per parameter.
#' @param tau Scalar prior precision.
#' @param n_training Integer, number of training points.
#' @param c Integer, number of classes (logit outputs).
#' @return An (M x C) numeric matrix of per-sample, per-class logit
#'   variances.
#' @noRd
compute_logit_variances <- function(features, h_diag, tau, n_training, c) {
  d <- ncol(features)
  m <- nrow(features)

  var_logits <- matrix(NA_real_, nrow = m, ncol = c)

  for (cl in seq_len(c)) {
    offset <- (cl - 1L) * (d + 1L)
    h_w <- h_diag[offset + seq_len(d)]
    h_b <- h_diag[offset + d + 1L]

    prec_w <- n_training * h_w + tau
    prec_b <- n_training * h_b + tau

    var_w <- 1 / prec_w
    var_b <- 1 / prec_b

    var_logits[, cl] <- as.vector(features^2 %*% var_w) + var_b
  }

  var_logits
}


#' MC Sampling for Classification Confidence Intervals
#'
#' @description
#' Samples K times from the Laplace posterior over logits, applies
#' sigmoid (binary) or softmax (multi-class), and returns per-class
#' quantiles of the resulting probability samples.
#'
#' @details
#' This captures epistemic uncertainty only — the posterior variance of
#' `P(Y|X)` under the Laplace approximation to the weight posterior.
#'
#' @param combined_model A Keras model with outputs `logits` and
#'   `features`.
#' @param x Processed predictor data (matrix or array).
#' @param h_diag Diagonal GGN entries (length `C * (d + 1)`).
#' @param tau Scalar prior precision.
#' @param n_training Integer, number of training points.
#' @param num_classes Integer, number of logit output units (1 for sigmoid
#'   binary, C for softmax multi-class).
#' @param lvl Character vector of class level names.
#' @param level Confidence level (e.g. 0.95).
#' @param n_samples Integer, number of MC draws (default 1000).
#' @return A data frame with per-class `.pred_lower_Level` and
#'   `.pred_upper_Level` columns.
#' @noRd
sample_conf_int_cls <- function(
  combined_model,
  x,
  h_diag,
  tau,
  n_training,
  num_classes,
  lvl,
  level,
  n_samples = 1000L
) {
  # Combined model outputs logits and features in one Keras forward pass
  combined_pred <- predict(combined_model, x)
  logits <- as.matrix(combined_pred$logits)
  features <- as.matrix(combined_pred$features)

  m <- nrow(features)
  c <- num_classes
  var_logits <- compute_logit_variances(features, h_diag, tau, n_training, c)

  # GPU-side MC sampling using Keras eager ops
  logits_k <- keras3::op_convert_to_tensor(logits, dtype = "float32")
  var_k <- keras3::op_convert_to_tensor(var_logits, dtype = "float32")

  samples <- keras3::random_normal(
    shape = as.integer(c(m, c, n_samples)),
    mean = keras3::op_expand_dims(logits_k, 3L),
    stddev = keras3::op_sqrt(
      keras3::op_maximum(keras3::op_expand_dims(var_k, 3L), 0)
    )
  )

  probs <- keras3::op_softmax(samples, axis = 2L)

  lo_prob <- (1 - level) / 2
  hi_prob <- 1 - lo_prob
  lo <- as.array(keras3::op_quantile(probs, lo_prob, axis = 3L))
  hi <- as.array(keras3::op_quantile(probs, hi_prob, axis = 3L))

  result_cols <- list()
  for (cl in seq_len(c)) {
    result_cols[[paste0(".pred_lower_", lvl[cl])]] <- lo[, cl]
    result_cols[[paste0(".pred_upper_", lvl[cl])]] <- hi[, cl]
  }

  as.data.frame(result_cols)
}


#' MC Sampling for Classification Prediction Intervals
#'
#' @description
#' Samples K times from the Laplace posterior, transforms to probability
#' scale, then draws class labels from `Bernoulli(p)` or `Categorical(p)`,
#' and returns per-class quantiles of the resulting indicator samples.
#'
#' @details
#' This captures both epistemic and aleatoric uncertainty — matching the
#' posterior predictive behaviour of Stan (`posterior_predict()`) and BART
#' (`type = "ppd"`) classification engines.
#'
#' @inheritParams sample_conf_int_cls
#' @return A data frame with per-class `.pred_lower_Level` and
#'   `.pred_upper_Level` columns.
#' @noRd
sample_pred_int_cls <- function(
  combined_model,
  x,
  h_diag,
  tau,
  n_training,
  num_classes,
  lvl,
  level,
  n_samples = 1000L
) {
  combined_pred <- predict(combined_model, x)
  logits <- as.matrix(combined_pred$logits)
  features <- as.matrix(combined_pred$features)

  m <- nrow(features)
  c <- num_classes
  var_logits <- compute_logit_variances(features, h_diag, tau, n_training, c)

  # GPU-side MC sampling with categorical draws
  logits_k <- keras3::op_convert_to_tensor(logits, dtype = "float32")
  var_k <- keras3::op_convert_to_tensor(var_logits, dtype = "float32")

  samples <- keras3::random_normal(
    shape = as.integer(c(m, c, n_samples)),
    mean = keras3::op_expand_dims(logits_k, 3L),
    stddev = keras3::op_sqrt(
      keras3::op_maximum(keras3::op_expand_dims(var_k, 3L), 0)
    )
  )

  # Draw one categorical label per sample using random_categorical
  # Reshape (M, C, n_samples) → (M * n_samples, C) for batch categorical draw
  logits_2d <- keras3::op_reshape(samples, as.integer(c(m * n_samples, c)))
  classes <- keras3::random_categorical(logits_2d, num_samples = 1L)
  indicators_2d <- keras3::op_one_hot(
    keras3::op_squeeze(classes, 2L),
    as.integer(c)
  )
  indicators <- keras3::op_reshape(
    indicators_2d,
    as.integer(c(m, c, n_samples))
  )

  lo_prob <- (1 - level) / 2
  hi_prob <- 1 - lo_prob
  lo <- as.array(keras3::op_quantile(indicators, lo_prob, axis = 3L))
  hi <- as.array(keras3::op_quantile(indicators, hi_prob, axis = 3L))

  result_cols <- list()
  for (cl in seq_len(c)) {
    result_cols[[paste0(".pred_lower_", lvl[cl])]] <- lo[, cl]
    result_cols[[paste0(".pred_upper_", lvl[cl])]] <- hi[, cl]
  }

  as.data.frame(result_cols)
}


#' Predict Confidence Intervals for Classification via LLA
#'
#' @description
#' Predict-time entry point for classification confidence intervals.
#' Called by parsnip via
#' `c(pkg = "kerasnip", fun = "laplace_conf_int_cls")`.
#'
#' For each output, this uses Monte Carlo sampling from the Laplace
#' posterior over logits, transforms to probability scale via sigmoid /
#' softmax, and returns per-class quantile-based intervals.
#'
#' @param object The raw Keras model (from `object$fit$fit`).
#' @param x Processed predictor data (matrix or array).
#' @param laplace_data A named list of Laplace posterior data (from
#'   `object$fit$laplace`).
#' @param lvl Character vector of class level names (from
#'   `object$fit$lvl`).
#' @param level Confidence level (default 0.95).
#' @return For single-output: a data frame with per-class
#'   `.pred_lower_Level` and `.pred_upper_Level` columns.
#' @export
#' @keywords internal
laplace_conf_int_cls <- function(
  object,
  x,
  laplace_data,
  lvl,
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
    sample_conf_int_cls(
      combined_model = entry$combined_model,
      x = x,
      h_diag = entry$h_diag,
      tau = entry$tau,
      n_training = entry$n_training,
      num_classes = entry$num_classes,
      lvl = lvl,
      level = level,
      n_samples = 1000L
    )
  })
}


#' Predict Prediction Intervals for Classification via LLA
#'
#' @description
#' Predict-time entry point for classification prediction intervals.
#' Called by parsnip via
#' `c(pkg = "kerasnip", fun = "laplace_pred_int_cls")`.
#'
#' After computing epistemic probability samples (as in conf_int), this
#' draws class labels from `Bernoulli(p_sample)` or
#' `Categorical(p_sample)` and returns per-class quantiles of the
#' resulting indicator samples — matching the posterior predictive
#' behaviour of Stan and BART classification engines.
#'
#' @inheritParams laplace_conf_int_cls
#' @return For single-output: a data frame with per-class
#'   `.pred_lower_Level` and `.pred_upper_Level` columns.
#' @export
#' @keywords internal
laplace_pred_int_cls <- function(
  object,
  x,
  laplace_data,
  lvl,
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
    sample_pred_int_cls(
      combined_model = entry$combined_model,
      x = x,
      h_diag = entry$h_diag,
      tau = entry$tau,
      n_training = entry$n_training,
      num_classes = entry$num_classes,
      lvl = lvl,
      level = level,
      n_samples = 1000L
    )
  })
}


#' Post-Process Classification Interval Results into a Tibble
#'
#' @description
#' Formats raw classification interval prediction results into a tibble.
#' The results are already a data frame with correctly-named per-class
#' `.pred_lower_Level` and `.pred_upper_Level` columns.
#'
#' @param results A data frame (single-output) or a named list of data
#'   frames (multi-output) with per-class interval columns.
#' @param object The `model_fit` object (unused; required by parsnip
#'   convention).
#' @return A tibble.
#' @noRd
postprocess_intervals_cls <- function(results, object) {
  if (
    is.list(results) &&
      !is.null(names(results)) &&
      !inherits(results, "data.frame")
  ) {
    is_multi <- length(results) > 1L
    combined_preds <- lapply(names(results), function(nm) {
      df <- results[[nm]]
      if (is_multi) {
        colnames(df) <- paste0(colnames(df), "_", nm)
      }
      tibble::as_tibble(df)
    })
    do.call(tibble::tibble, combined_preds)
  } else {
    if (is.list(results)) {
      df <- results[[1L]]
    } else {
      df <- results
    }
    tibble::as_tibble(df)
  }
}
