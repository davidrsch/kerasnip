#' Evaluate a Fitted Kerasnip Model on New Data
#'
#' @title Evaluate a Kerasnip Model
#' @description
#' This function provides an `kera_evaluate()` method for `model_fit` objects
#' created by `kerasnip`. It preprocesses the new data into the format expected
#' by Keras and then calls `keras3::evaluate()` on the underlying model to
#' compute the loss and any other metrics.
#'
#' @param object A `model_fit` object produced by a `kerasnip` specification.
#' @param x A data frame or matrix of new predictor data.
#' @param y A vector or data frame of new outcome data corresponding to `x`.
#' @param ... Additional arguments passed on to `keras3::evaluate()` (e.g.,
#'   `batch_size`).
#'
#' @return A named list containing the evaluation results (e.g., `loss`,
#'   `accuracy`). The names are determined by the metrics the model was compiled
#'   with.
#'
#' @examples
#' \donttest{
#' if (requireNamespace("keras3", quietly = TRUE)) {
#' library(keras3)
#' library(parsnip)
#'
#' # 1. Define layer blocks
#' input_block <- function(model, input_shape) {
#'   keras_model_sequential(input_shape = input_shape)
#' }
#' hidden_block <- function(model, units = 32) {
#'   model |> layer_dense(units = units, activation = "relu")
#' }
#' output_block <- function(model, num_classes) {
#'   model |> layer_dense(units = num_classes, activation = "softmax")
#' }
#'
#' # 2. Define and fit a model ----
#' create_keras_sequential_spec(
#'   model_name = "my_mlp_tools",
#'   layer_blocks = list(
#'     input = input_block,
#'     hidden = hidden_block,
#'     output = output_block
#'   ),
#'   mode = "classification"
#' )
#'
#' mlp_spec <- my_mlp_tools(
#'   hidden_units = 32,
#'   compile_loss = "categorical_crossentropy",
#'   compile_optimizer = "adam",
#'   compile_metrics = "accuracy",
#'   fit_epochs = 5
#' ) |> set_engine("keras")
#'
#' x_train <- matrix(rnorm(100 * 10), ncol = 10)
#' y_train <- factor(sample(0:1, 100, replace = TRUE))
#' train_df <- data.frame(x = I(x_train), y = y_train)
#'
#' fitted_mlp <- fit(mlp_spec, y ~ x, data = train_df)
#'
#' # 3. Evaluate the model on new data ----
#' x_test <- matrix(rnorm(50 * 10), ncol = 10)
#' y_test <- factor(sample(0:1, 50, replace = TRUE))
#'
#' eval_metrics <- keras_evaluate(fitted_mlp, x_test, y_test)
#' print(eval_metrics)
#'
#' # 4. Extract the Keras model object ----
#' keras_model <- extract_keras_model(fitted_mlp)
#' summary(keras_model)
#'
#' # 5. Extract the training history ----
#' history <- extract_keras_history(fitted_mlp)
#' plot(history)
#' remove_keras_spec("my_mlp_tools")
#' }
#' }
#' @export
keras_evaluate <- function(object, x, y = NULL, ...) {
  # 1. Get the correct processing functions from the fit object
  process_x_fun <- object$fit$process_x
  process_y_fun <- object$fit$process_y

  if (is.null(process_x_fun) || is.null(process_y_fun)) {
    stop(
      "Could not find processing functions in the model fit object. ",
      "Please ensure the model was fitted with a recent version of kerasnip.",
      call. = FALSE
    )
  }

  # 2. Preprocess predictor data (x)
  x_processed <- process_x_fun(x)
  x_proc <- x_processed$x_proc

  # 3. Preprocess outcome data (y)
  y_proc <- NULL
  if (!is.null(y)) {
    # Note: For evaluation, we pass the class levels from the trained model
    # to ensure consistent encoding of the new data.
    y_processed <- process_y_fun(
      y,
      is_classification = !is.null(object$fit$lvl),
      class_levels = object$fit$lvl
    )
    y_proc <- y_processed$y_proc
  }

  # 4. Call the underlying Keras evaluate method
  keras_model <- tryCatch(
    {
      reticulate::py_validate_xptr(object$fit$fit)
      object$fit$fit
    },
    error = function(e) {
      if (is.null(object$fit$keras_bytes)) {
        stop(e)
      }
      keras_model_from_bytes(object$fit$keras_bytes)
    }
  )
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
  tryCatch(
    {
      reticulate::py_validate_xptr(object$fit$fit)
      object$fit$fit
    },
    error = function(e) {
      if (is.null(object$fit$keras_bytes)) {
        stop(e)
      }
      keras_model_from_bytes(object$fit$keras_bytes)
    }
  )
}

#' Extract Keras Training History
#'
#' @title Extract Keras Training History
#' @description
#' Extracts and returns the training history from a `parsnip` `model_fit` object
#' created by `kerasnip`.
#'
#' @details
#' The history object contains the metrics recorded during model training, such
#' as loss and accuracy, for each epoch. This is highly useful for visualizing
#' the training process and diagnosing issues like overfitting.
#' The returned object can be plotted directly.
#'
#' @param object A `model_fit` object produced by a `kerasnip` specification.
#' @return A `keras_training_history` object. You can call `plot()` on this
#'   object to visualize the learning curves.
#' @seealso keras_evaluate, extract_keras_model
#' @export
extract_keras_history <- function(object) {
  object$fit$history
}

#' Tidy a Fitted Kerasnip Model
#'
#' @description
#' Returns a tibble with one row per layer of the underlying Keras model,
#' summarising the layer name, Python class, and parameter count.
#'
#' @param x A `kerasnip_model_fit` object.
#' @param ... Not used.
#' @return A tibble with columns `layer` (character), `class` (character),
#'   and `n_params` (integer).
#' @importFrom generics tidy
#' @importFrom tibble tibble
#' @export
tidy.kerasnip_model_fit <- function(x, ...) {
  model <- extract_keras_model(x)
  layers <- model$layers

  layer_name <- vapply(layers, function(l) l$name, character(1))
  layer_class <- vapply(layers, function(l) {
    parts <- strsplit(class(l)[[1L]], ".", fixed = TRUE)[[1L]]
    tail(parts, 1L)
  }, character(1))
  n_params <- vapply(layers, function(l) {
    tryCatch(as.integer(l$count_params()), error = function(e) NA_integer_)
  }, integer(1))

  tibble(layer = layer_name, class = layer_class, n_params = n_params)
}

#' Glance at a Fitted Kerasnip Model
#'
#' @description
#' Returns a one-row tibble of summary statistics from the final training epoch:
#' every metric the model was compiled with (e.g. `loss`, `accuracy`).
#'
#' @param x A `kerasnip_model_fit` object.
#' @param ... Not used.
#' @return A one-row tibble with one column per compiled metric. Returns an
#'   empty tibble if training history has been stripped (e.g. by butcher).
#' @importFrom generics glance
#' @importFrom tibble as_tibble tibble
#' @export
glance.kerasnip_model_fit <- function(x, ...) {
  history <- x$fit$history
  if (is.null(history) || is.null(history$metrics)) {
    return(tibble::tibble())
  }
  last_vals <- lapply(history$metrics, function(m) m[[length(m)]])
  as_tibble(last_vals)
}
