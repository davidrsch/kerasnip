#' Evaluate a Kerasnip Model
#'
#' This function provides an `kera_evaluate()` method for `model_fit` objects
#' created by `kerasnip`. It preprocesses the data into the format expected by
#' Keras and then calls `keras3::evaluate()` on the underlying model.
#'
#' @param object A `model_fit` object produced by a `kerasnip` specification.
#' @param x A data frame or matrix of predictors.
#' @param y A vector or data frame of outcomes.
#' @param ... Additional arguments passed on to `keras3::evaluate()`.
#'
#' @return A `list` with evaluation results
#'
#' @export
keras_evaluate <- function(object, x, y = NULL, ...) {
  # 1. Preprocess predictor data (x)
  x_processed <- process_x(x)
  x_proc <- x_processed$x_proc

  # 2. Preprocess outcome data (y)
  y_proc <- NULL
  if (!is.null(y)) {
    y_processed <- process_y(
      y,
      is_classification = !is.null(object$fit$lvl),
      class_levels = object$fit$lvl
    )
    y_proc <- y_processed$y_proc
  }

  # 3. Call the underlying Keras evaluate method
  keras_model <- object$fit$fit
  keras3::evaluate(keras_model, x = x_proc, y = y_proc, ...)
}

#' Extract the Raw Keras Model from a Kerasnip Fit
#'
#' @title Extract Keras Model from a Fitted Kerasnip Object
#' @description
#' Extracts and returns the underlying Keras model object from a `parsnip`
#' `model_fit` object created by `kerasnip`.
#'
#' @details
#' This is useful when you need to work directly with the Keras model object for
#' tasks like inspecting layer weights, creating custom plots, or passing it to
#' other Keras-specific functions.
#'
#' @param object A `model_fit` object produced by a `kerasnip` specification.
#'
#' @return The raw Keras model object (`keras_model`).
#' @seealso keras_evaluate, extract_keras_history
#' @export
extract_keras_model <- function(object) {
  object$fit$fit
}

#' Extract Keras Training History
#'
#' @description
#' Extracts and returns the training history of a Keras model fitted with `kerasnip`.
#'
#' @param object A `model_fit` object produced by a `kerasnip` specification.
#' @return A `keras_training_history` containing the training history (metrics per epoch).
#' @export
extract_keras_history <- function(object) {
  object$fit$history
}
