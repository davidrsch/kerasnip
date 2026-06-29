# Last-Layer Laplace Approximation for kerasnip
#
# Internal functions implementing the diagonal Generalised Gauss-Newton (GGN)
# approximation to the Hessian, marginal-likelihood optimisation of prior
# precision (tau) and observation noise (sigma_sq), and per-sample epistemic /
# predictive variance computation.
#
# Reference: Daxberger, E. et al. (2021). "Laplace Redux -- Effortless
#   Bayesian Deep Learning." NeurIPS 34, 20089-20103.
#   https://arxiv.org/abs/2106.14806
#
# Following Daxberger et al. (2021), the Laplace posterior
# q(w) = N(w_MAP, (H + tau*I)^{-1}) is fitted over the last-layer weights of a
# trained model.  The diagonal of the GGN is used as H.  tau and sigma_sq_noise
# are jointly optimised via the marginal likelihood (empirical Bayes).
#
# Regression models with a Dense output layer are supported.  Classification
# support (binary and multi-class) is provided in
# laplace_approx_classification.R, using the cross-entropy GGN diagonal.
#
# All functions are internal (not exported).  The single public entry point is
# laplace_all_regression(), called from the fit engines.

#' Discover Output and Penultimate Layers
#'
#' @description
#' Scans `model$layers` to find the output Dense layer and the penultimate
#' computational layer feeding into it.  Avoids tensor-walking, which is
#' unreliable across Keras 3 model types because KerasTensors do not expose
#' a `.layer` attribute and Sequential model wrappers do not define
#' `.output` until built.
#'
#' @param model A compiled Keras model.
#' @return A named list keyed by output name, each entry containing
#'   `output_layer_name` and `penultimate_layer_name`.  Returns `NULL` if
#'   no usable Dense output is found (with a warning).
#' @noRd
find_output_layer_infos <- function(model) {
  layers <- model$layers

  # Identify all Dense layers
  is_dense <- vapply(
    layers,
    function(l) {
      inherits(l, "keras.src.layers.core.dense.Dense")
    },
    logical(1L)
  )

  if (!any(is_dense)) {
    warning(
      "No Dense layer found in model. ",
      "Laplace approximation requires a Dense output layer.",
      call. = FALSE
    )
    return(NULL)
  }

  dense_indices <- which(is_dense)

  # Detect multi-output via model$output (works for functional models; returns
  # NULL for sequential, which is fine because sequential is single-output).
  model_output <- tryCatch(model$output, error = function(e) NULL)
  if (is.list(model_output) && !is.null(names(model_output))) {
    return(find_multi_output_layer_infos(layers, model_output))
  }

  # --- Single-output path ---
  last_dense_idx <- dense_indices[length(dense_indices)]
  output_layer <- layers[[last_dense_idx]]

  # Find penultimate layer: walk backward from the output Dense, skipping
  # InputLayer, Sequential wrapper, and other non-computational layers
  penultimate_idx <- last_dense_idx - 1L
  while (penultimate_idx >= 1L) {
    layer_class <- class(layers[[penultimate_idx]])[1L]
    is_skip <- grepl("input_layer|InputLayer|Sequential", layer_class)
    if (!is_skip) {
      break
    }
    penultimate_idx <- penultimate_idx - 1L
  }

  if (penultimate_idx < 1L) {
    warning(
      "No penultimate computational layer found before the output Dense ",
      "layer. The model may have no hidden layers.",
      call. = FALSE
    )
    return(NULL)
  }

  penultimate_layer <- layers[[penultimate_idx]]

  stats::setNames(
    list(list(
      output_layer_name = output_layer$name,
      penultimate_layer_name = penultimate_layer$name
    )),
    "output"
  )
}


#' Discover Layers for Multi-Output Models
#'
#' @description
#' For each named output in a functional Keras model, finds the matching
#' Dense output layer and its penultimate computational layer.  Used by
#' `find_output_layer_infos()` when `model$output` returns a named list.
#'
#' @param layers The flat list of layers from `model$layers`.
#' @param model_output The named list from `model$output` (functional API
#'   only).
#' @return A named list with one entry per output, each containing
#'   `output_layer_name` and `penultimate_layer_name`.  Returns `NULL` if
#'   no matching Dense layers are found.
#' @noRd
find_multi_output_layer_infos <- function(layers, model_output) {
  # Build a lookup of output tensor names to their producing layers
  tensor_to_layer <- list()
  for (l in layers) {
    l_out <- tryCatch(l$output, error = function(e) NULL)
    if (!is.null(l_out) && !is.null(l_out$name)) {
      tensor_to_layer[[l_out$name]] <- l
    }
  }

  result <- lapply(names(model_output), function(nm) {
    # Find the Dense layer that produced this output tensor
    output_tensor_name <- model_output[[nm]]$name
    out_layer <- tensor_to_layer[[output_tensor_name]]

    if (
      is.null(out_layer) ||
        !inherits(out_layer, "keras.src.layers.core.dense.Dense")
    ) {
      return(NULL)
    }

    # Trace penultimate: tensor feeding into the output Dense
    input_tensor_name <- out_layer$input$name
    penultimate_layer <- tensor_to_layer[[input_tensor_name]]

    list(
      output_layer_name = out_layer$name,
      penultimate_layer_name = if (is.null(penultimate_layer)) {
        NULL
      } else {
        penultimate_layer$name
      }
    )
  })

  names(result) <- names(model_output)
  result <- result[!vapply(result, is.null, logical(1L))]
  if (length(result) == 0L) NULL else result
}


#' Get Model Input Tensor Across Keras 3 APIs
#'
#' @description
#' Functional models expose `model$input` directly; Sequential wrappers
#' do not, so `model$layers[[1]]$input` is used as fallback.
#'
#' @param model A compiled Keras model.
#' @return The model's input tensor.
#' @noRd
get_model_input <- function(model) {
  # Functional models expose $input directly; sequential wrappers don't
  # but their first layer does
  tryCatch(model$input, error = function(e) model$layers[[1L]]$input)
}


# Single-output: class_levels is NULL (regression) or character vector.
# Multi-output:  class_levels is a named list of NULLs (regression)
#                or a named list of character vectors (classification).
#' @noRd
is_regression_mode <- function(class_levels) {
  is.null(class_levels) ||
    (is.list(class_levels) && all(vapply(class_levels, is.null, logical(1L))))
}


#' Compute Diagonal GGN for MSE Loss
#'
#' @description
#' For a linear last layer with MSE loss, the diagonal of the Generalised
#' Gauss-Newton approximation to the Hessian is simply the second moment of
#' the feature activations plus 1 for the bias term.
#'
#' @param features An (N x d) numeric matrix of penultimate-layer activations.
#' @return A numeric vector of length (d + 1): column means of `features^2`
#'   followed by `1` (the bias diagonal entry, since h_b = 1 for MSE GGN).
#' @noRd
h_diag_regression <- function(features) {
  c(colMeans(features^2), 1)
}


#' Negative Log Marginal Likelihood (Diagonal Laplace)
#'
#' @description
#' Evaluates the negative log marginal likelihood under the diagonal Laplace
#' approximation.  Used as the objective for hyperparameter optimisation.
#'
#' @details
#' Formula (constant `-N/2 log(2*pi)` omitted; it does not affect the optimum):
#' ```
#' nll = N/2 * log(sigma_sq) + RSS / (2 * sigma_sq)
#'     - D/2 * log(tau) + tau * w_sq / 2
#'     + 1/2 * sum(log(N * h_j / sigma_sq + tau))
#' ```
#'
#' @param log_params Numeric vector of length 2: `c(log(tau), log(sigma_sq))`.
#'   Log-space ensures positivity without constrained optimisation.
#' @param h_diag Numeric vector of length D (d_weights + 1 for bias).
#'   Diagonal of the GGN per parameter.
#' @param rss Scalar. Residual sum of squares at the MAP estimate.
#' @param w_sq Scalar. Sum of squared MAP weights (including bias).
#' @param n Integer. Number of training points.
#' @param d Integer. Total number of last-layer parameters.
#' @return A scalar: the negative log marginal likelihood.
#' @noRd
neg_log_ml_regression <- function(
  log_params,
  h_diag,
  rss,
  w_sq,
  n,
  d
) {
  tau <- exp(log_params[1L])
  sigma_sq <- exp(log_params[2L])

  # Posterior precision diagonal: H_jj = N * h_j / sigma_sq + tau
  new_h_diag <- n * h_diag / sigma_sq + tau

  nll <- (n / 2) *
    log(sigma_sq) +
    rss / (2 * sigma_sq) -
    (d / 2) * log(tau) +
    tau * w_sq / 2 +
    0.5 * sum(log(new_h_diag))

  nll
}


#' Optimise Laplace Hyperparameters
#'
#' @description
#' Optimises the prior precision `tau` and observation noise `sigma_sq_noise`
#' by minimising the negative log marginal likelihood via Nelder-Mead in
#' log-space (unconstrained, 2-parameter problem).
#'
#' @param h_diag,rss,w_sq,n,d Passed through to
#'   `neg_log_ml_regression()`.
#' @return A list with elements `tau` and `sigma_sq_noise` (both scalars).
#' @noRd
optim_laplace_regression <- function(h_diag, rss, w_sq, n, d) {
  sigma_sq_init <- max(rss / n, 1e-8)
  tau_init <- 1 / (sigma_sq_init * d)

  result <- stats::optim(
    par = c(log(tau_init), log(sigma_sq_init)),
    fn = neg_log_ml_regression,
    h_diag = h_diag,
    rss = rss,
    w_sq = w_sq,
    n = n,
    d = d,
    method = "Nelder-Mead"
  )

  list(
    tau = exp(result$par[1L]),
    sigma_sq_noise = exp(result$par[2L])
  )
}


#' Compute Laplace Posterior for One Regression Output
#'
#' @description
#' Builds a combined Keras model (input -> `list(pred, features)`) that
#' returns both the point prediction and the penultimate-layer features in
#' a single forward pass.  Computes the diagonal GGN, optimises `tau` and
#' `sigma_sq_noise` via the marginal likelihood, and serialises the
#' combined model for cross-session survival.
#'
#' @param model A compiled, fitted Keras model.
#' @param x_proc Processed predictor data (matrix or array).
#' @param y_mat_one Processed outcome data for one output (matrix).
#' @param layer_info A list with `output_layer_name` and
#'   `penultimate_layer_name`.
#' @return A list with `h_diag`, `tau`, `sigma_sq_noise`, `n_training`,
#'   `combined_model`, and `combined_model_bytes`.
#' @noRd
laplace_one_regression <- function(
  model,
  x_proc,
  y_mat_one,
  layer_info
) {
  output_layer <- model$get_layer(layer_info$output_layer_name)
  penultimate_layer <- model$get_layer(layer_info$penultimate_layer_name)
  model_input <- get_model_input(model)

  combined_model <- keras3::keras_model(
    inputs = model_input,
    outputs = list(
      pred = output_layer$output,
      features = penultimate_layer$output
    )
  )

  combined_pred <- predict(combined_model, x_proc)
  y_pred <- as.vector(combined_pred$pred)
  features <- as.matrix(combined_pred$features)

  # MAP weights (kernel + bias) from the output Dense layer
  weights_list <- output_layer$get_weights()
  w_mat <- weights_list[[1L]]
  b_vec <- weights_list[[2L]]
  w_sq <- sum(as.vector(w_mat)^2) + sum(as.vector(b_vec)^2)

  # Residuals and RSS
  y_mat_vec <- as.vector(y_mat_one)
  residuals <- y_mat_vec - y_pred
  rss <- sum(residuals^2)

  n <- nrow(features)
  h_diag <- h_diag_regression(features)
  d <- length(h_diag)

  hp <- optim_laplace_regression(h_diag, rss, w_sq, n, d)

  combined_model_bytes <- keras_model_to_bytes(combined_model)

  list(
    h_diag = h_diag,
    tau = hp$tau,
    sigma_sq_noise = hp$sigma_sq_noise,
    n_training = n,
    combined_model = combined_model,
    combined_model_bytes = combined_model_bytes
  )
}


#' Compute Laplace Posterior for All Regression Outputs
#'
#' @description
#' Public orchestrator called from [generic_sequential_fit()] and
#' [generic_functional_fit()] after `keras3::fit()`.  For each regression
#' output with a Dense last layer, this function computes the diagonal GGN,
#' optimises the prior precision `tau` and observation noise `sigma_sq_noise`
#' via the marginal likelihood, and stores a combined Keras model that returns
#' both predictions and penultimate-layer features in a single forward pass.
#'
#' @details
#' `conf_int` and `pred_int` carry specific statistical meanings in
#' tidymodels.  `conf_int` is uncertainty on `E[Y|X]` (the conditional
#' mean) and `pred_int` is uncertainty on a new observation `Y|X`.
#' Regression uses analytic Normal-based intervals.  Classification
#' support (binary and multi-class) is provided in
#' `laplace_approx_classification.R`, using MC sampling from the Laplace
#' posterior over logits with per-class quantile intervals.
#'
#' @param model A compiled, fitted Keras model.
#' @param x_proc Processed predictor data (matrix or array).
#' @param y_mat Processed outcome data (matrix for single-output, or named
#'   list of matrices for multi-output).
#' @return A named list with one entry per output, each containing `h_diag`,
#'   `tau`, `sigma_sq_noise`, `n_training`, `combined_model`, and
#'   `combined_model_bytes`.  Returns `NULL` when no Dense output layer is
#'   found (with a warning).
#' @noRd
laplace_all_regression <- function(model, x_proc, y_mat) {
  layer_infos <- find_output_layer_infos(model)
  if (is.null(layer_infos)) {
    return(NULL)
  }

  is_multi <- is.list(y_mat) && !is.null(names(y_mat))

  if (is_multi) {
    result <- lapply(names(layer_infos), function(nm) {
      laplace_one_regression(
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
    result <- laplace_one_regression(
      model,
      x_proc,
      y_mat_one = y_mat,
      layer_info = info
    )
    stats::setNames(list(result), names(layer_infos)[1L])
  }
}
