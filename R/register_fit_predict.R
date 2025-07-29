#' Register Fit and Prediction Methods with Parsnip
#'
#' Defines how to fit the custom Keras model and how to generate predictions
#' for both regression and classification modes. It links the model to the
#' generic fitting implementation (`generic_sequential_fit`) and sets up
#' the appropriate prediction post-processing.
#'
#' @param model_name The name of the new model.
#' @param mode The model mode ("regression" or "classification").
#' @param layer_blocks The named list of layer block functions, which is passed
#'   as a default argument to the fit function.
#' @return Invisibly returns `NULL`. Called for its side effects.
#' @noRd
register_fit_predict <- function(model_name, mode, layer_blocks) {
  # Fit method
  parsnip::set_fit(
    model = model_name,
    eng = "keras",
    mode = mode,
    value = list(
      interface = "data.frame",
      protect = c("x", "y"),
      func = c(pkg = "kerasnip", fun = "generic_sequential_fit"),
      defaults = list(layer_blocks = layer_blocks)
    )
  )

  # Regression prediction
  if (mode == "regression") {
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "regression",
      type = "numeric",
      value = list(
        pre = NULL,
        post = keras_postprocess_numeric,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = rlang::expr(as.matrix(new_data))
        )
      )
    )
  } else {
    # Classification predictions
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "classification",
      type = "class",
      value = list(
        pre = NULL,
        post = keras_postprocess_classes,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = rlang::expr(as.matrix(new_data))
        )
      )
    )
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "classification",
      type = "prob",
      value = list(
        pre = NULL,
        post = keras_postprocess_probs,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = rlang::expr(as.matrix(new_data))
        )
      )
    )
  }
}

#' Post-process Keras Numeric Predictions
#'
#' Formats raw numeric predictions from a Keras model into a tibble with a
#' standardized `.pred` column.
#'
#' @param results A matrix of numeric predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with a `.pred` column.
#' @noRd
keras_postprocess_numeric <- function(results, object) {
  tibble::tibble(.pred = as.vector(results))
}

#' Post-process Keras Probability Predictions
#'
#' Formats raw probability predictions from a Keras model into a tibble
#' with class-specific column names.
#'
#' @param results A matrix of probability predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with named columns for each class probability.
#' @noRd
keras_postprocess_probs <- function(results, object) {
  # The levels are now nested inside the fit object
  colnames(results) <- object$fit$lvl
  tibble::as_tibble(results)
}

#' Post-process Keras Class Predictions
#'
#' Converts raw probability predictions from a Keras model into factor-based
#' class predictions.
#'
#' @param results A matrix of probability predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with a `.pred_class` column containing factor predictions.
#' @noRd
keras_postprocess_classes <- function(results, object) {
  # The levels are now nested inside the fit object
  lvls <- object$fit$lvl
  if (ncol(results) == 1) {
    # Binary classification
    pred_class <- ifelse(results[, 1] > 0.5, lvls[2], lvls[1])
    pred_class <- factor(pred_class, levels = lvls)
  } else {
    # Multiclass classification
    pred_class_int <- apply(results, 1, which.max)
    pred_class <- lvls[pred_class_int]
    pred_class <- factor(pred_class, levels = lvls)
  }
  tibble::tibble(.pred_class = pred_class)
}