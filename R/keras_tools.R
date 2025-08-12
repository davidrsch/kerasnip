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
