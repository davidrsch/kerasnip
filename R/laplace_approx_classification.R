# Last-Layer Laplace Approximation for Classification
#
# Parallel to laplace_approx_regression.R but using the Generalised Gauss-Newton
# (GGN) diagonal for cross-entropy / softmax output layers.  The Bernoulli /
# multinomial likelihood has fixed scale (no sigma_sq noise parameter), so the
# marginal likelihood optimises only the prior precision tau.
#
# Reference: Daxberger, E. et al. (2021). "Laplace Redux -- Effortless
#   Bayesian Deep Learning." NeurIPS 34, 20089-20103.
#   https://arxiv.org/abs/2106.14806
#
# Following Daxberger et al. (2021), Section 4.2.
#
# All functions are internal (not exported).  The public entry point is
# laplace_all_classification(), called from the fit engines.

#' Compute Diagonal GGN for Cross-Entropy / Softmax
#'
#' @description
#' For a last layer with softmax (or sigmoid) output and cross-entropy loss,
#' the diagonal of the GGN includes a `p_c * (1-p_c)` factor from the Hessian
#' of the NLL with respect to the logits.  Compare with
#' `h_diag_regression()` for MSE regression, where this factor is
#' always 1.
#'
#' @details
#' Per-class formula:
#' ```
#' h_jc = (1/N) * sum_i phi_ij^2 * p_ic * (1 - p_ic)   [weight W[j,c]]
#' h_bc = (1/N) * sum_i p_ic * (1 - p_ic)               [bias b[c]]
#' ```
#' Binary classification with sigmoid output (C = 1 logit) is a special case
#' of the same formula.
#'
#' @param features An (N x d) numeric matrix of penultimate-layer activations.
#' @param probs An (N x C) numeric matrix of class probabilities (sigmoid or
#'   softmax output).  For binary sigmoid with 1 output unit, probs has 1
#'   column with `p` values; the `1-p` complement is implicit.
#' @return A numeric vector of length `C * (d + 1)`, laid out per-class as
#'   `[w_1c, ..., w_dc, b_c]` for class 1, then class 2, etc.
#' @noRd
h_diag_classification <- function(features, probs) {
  n <- nrow(features)
  p_times_one_minus_p <- probs * (1 - probs)
  h_weights <- crossprod(features^2, p_times_one_minus_p) / n
  h_bias <- colMeans(p_times_one_minus_p)
  as.vector(rbind(h_weights, h_bias))
}


#' Negative Log Marginal Likelihood for Classification (Diagonal Laplace)
#'
#' @description
#' Evaluates the negative log marginal likelihood for classification under the
#' diagonal Laplace approximation.  Compared with the regression version this
#' has no `sigma_sq` parameter — the Bernoulli / multinomial scale is fixed.
#'
#' @details
#' Formula (constant terms omitted):
#' ```
#' nll = NLL_total - (D/2) * log(tau) + (tau/2) * w_sq
#'     + (1/2) * sum(log(N * h_j + tau))
#' ```
#' where `NLL_total` is the total cross-entropy summed over all training
#' points (not averaged).
#'
#' @param log_tau Scalar, log of the prior precision.
#' @param h_diag Numeric vector of length D.  Diagonal GGN entries per
#'   parameter as returned by `h_diag_classification()`.
#' @param nll_total Scalar.  Total cross-entropy loss over all N training
#'   points (sum, not mean).
#' @param w_sq Scalar.  Sum of squared MAP weights (including biases).
#' @param n Integer.  Number of training points.
#' @param d Integer.  Total number of last-layer parameters.
#' @return A scalar: the negative log marginal likelihood.
#' @noRd
neg_log_ml_classification <- function(
  log_tau,
  h_diag,
  nll_total,
  w_sq,
  n,
  d
) {
  tau <- exp(log_tau)

  # Posterior precision diagonal: H_jj = N * h_j + tau
  # (No sigma_sq in denominator — classification likelihood has fixed scale)
  h_post <- n * h_diag + tau

  nll <- nll_total -
    (d / 2) * log(tau) +
    tau * w_sq / 2 +
    0.5 * sum(log(h_post))

  nll
}


#' Optimise Laplace Prior Precision for Classification
#'
#' @description
#' Optimises the prior precision `tau` by minimising the negative log
#' marginal likelihood via Brent's method (`stats::optimize()`) in
#' log-space (single-parameter, unconstrained problem).  No
#' `sigma_sq_noise` is estimated — the Bernoulli / multinomial
#' likelihood has no separate scale parameter.
#'
#' @param h_diag,nll_total,w_sq,n,d Passed through to
#'   `neg_log_ml_classification()`.
#' @return A list with element `tau` (scalar).
#' @noRd
optim_laplace_classification <- function(
  h_diag,
  nll_total,
  w_sq,
  n,
  d
) {
  # Single-parameter: use Brent's method via optimize()
  tau_init <- max(d / w_sq, 1e-12)
  lo <- log(tau_init) - 10
  hi <- log(tau_init) + 10

  result <- stats::optimize(
    f = function(lt) {
      neg_log_ml_classification(lt, h_diag, nll_total, w_sq, n, d)
    },
    lower = lo,
    upper = hi
  )

  list(tau = exp(result$minimum))
}


#' Compute Laplace Posterior for One Classification Output
#'
#' @description
#' Builds a combined Keras model (input -> `list(logits, features)`) that
#' returns both logits and penultimate-layer features in a single forward
#' pass.  Computes the diagonal GGN for cross-entropy, optimises `tau` via
#' the marginal likelihood, and serialises the combined model.
#'
#' @param model A compiled, fitted Keras model.
#' @param x_proc Processed predictor data (matrix or array).
#' @param y_mat_one Processed outcome data for one output (one-hot matrix).
#' @param layer_info A list with `output_layer_name` and
#'   `penultimate_layer_name`.
#' @return A list with `h_diag`, `tau`, `n_training`, `num_classes`,
#'   `combined_model`, and `combined_model_bytes`.
#' @noRd
laplace_one_classification <- function(
  model,
  x_proc,
  y_mat_one,
  layer_info
) {
  output_layer <- model$get_layer(layer_info$output_layer_name)
  penultimate_layer <- model$get_layer(layer_info$penultimate_layer_name)
  model_input <- get_model_input(model)

  # Build a logit model using Keras ops: logits = phi @ kernel + bias
  # Avoids post-activation softmax from Dense(activation="softmax")
  logits_tensor <- keras3::op_add(
    keras3::op_matmul(penultimate_layer$output, output_layer$kernel),
    output_layer$bias
  )

  combined_model <- keras3::keras_model(
    inputs = model_input,
    outputs = list(
      logits = logits_tensor,
      features = penultimate_layer$output
    )
  )

  combined_pred <- predict(combined_model, x_proc)
  logits <- as.matrix(combined_pred$logits)
  features <- as.matrix(combined_pred$features)

  # Norm of MAP weights (use get_weights for R-native arrays)
  w <- output_layer$get_weights()
  w_sq <- sum(as.vector(w[[1L]])^2) + sum(as.vector(w[[2L]])^2)

  num_classes <- ncol(logits)

  logits_shifted <- logits - apply(logits, 1, max)
  exp_logits <- exp(logits_shifted)
  probs <- exp_logits / rowSums(exp_logits)

  y_mat_one <- as.matrix(y_mat_one)
  eps <- 1e-15
  nll_total <- -sum(y_mat_one * log(pmax(pmin(probs, 1 - eps), eps)))

  n_train <- nrow(features)
  h_diag <- h_diag_classification(features, probs)
  d_params <- length(h_diag)

  hp <- optim_laplace_classification(h_diag, nll_total, w_sq, n_train, d_params)

  combined_model_bytes <- keras_model_to_bytes(combined_model)

  list(
    h_diag = h_diag,
    tau = hp$tau,
    n_training = n_train,
    num_classes = num_classes,
    combined_model = combined_model,
    combined_model_bytes = combined_model_bytes
  )
}


#' Compute Laplace Posterior for All Classification Outputs
#'
#' @description
#' Public orchestrator called from [generic_sequential_fit()] and
#' [generic_functional_fit()] after `keras3::fit()`.  For each
#' classification output with a Dense last layer, this function computes
#' the diagonal GGN for the cross-entropy Hessian, optimises the prior
#' precision `tau` via the marginal likelihood, and stores a combined
#' Keras model that returns both logits and penultimate-layer features
#' in a single forward pass.
#'
#' @details
#' The stored `num_classes` field determines the number of softmax output
#' units used at predict time for interval back-transformation.
#'
#' @param model A compiled, fitted Keras model.
#' @param x_proc Processed predictor data (matrix or array).
#' @param y_mat Processed outcome data — one-hot encoded matrix for
#'   single-output classification, or a named list of such matrices for
#'   multi-output classification.
#' @return A named list with one entry per output, each containing
#'   `h_diag`, `tau`, `n_training`, `num_classes`,
#'   `combined_model`, and `combined_model_bytes`.  Returns `NULL` when
#'   no Dense output layer is found.
#' @noRd
laplace_all_classification <- function(model, x_proc, y_mat) {
  layer_infos <- find_output_layer_infos(model)
  if (is.null(layer_infos)) {
    return(NULL)
  }

  is_multi <- is.list(y_mat) && !is.null(names(y_mat))

  if (is_multi) {
    result <- lapply(names(layer_infos), function(nm) {
      laplace_one_classification(
        model,
        x_proc,
        y_mat_one = y_mat[[nm]],
        layer_info = layer_infos[[nm]]
      )
    })
    names(result) <- names(layer_infos)
    result
  } else {
    info <- layer_infos[[1L]]
    result <- laplace_one_classification(
      model,
      x_proc,
      y_mat_one = y_mat,
      layer_info = info
    )
    stats::setNames(list(result), names(layer_infos)[1L])
  }
}
