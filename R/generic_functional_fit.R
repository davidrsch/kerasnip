#' Generic Fitting Function for Functional Keras Models
#'
#' @title Internal Fitting Engine for Functional API Models
#' @description
#' This function serves as the internal engine for fitting `kerasnip` models
#' that are based on the Keras functional API. It is not intended to be called
#' directly by the user. The function is invoked by `parsnip::fit()` when a
#' `kerasnip` functional model specification is used.
#'
#' @details
#' The function orchestrates the three main steps of the model fitting process:
#' \enumerate{
#'   \item \strong{Build and Compile:} It calls
#'     `build_compile_func_model()` to construct the Keras model
#'     architecture based on the provided `layer_blocks` and hyperparameters.
#'   \item \strong{Process Data:} It preprocesses the input (`x`) and output
#'     (`y`) data into the format expected by Keras.
#'   \item \strong{Fit Model:} It calls `keras3::fit()` with the compiled model
#'     and processed data, passing along any fitting-specific arguments (e.g.,
#'     `epochs`, `batch_size`, `callbacks`).
#' }
#'
#' @param x A data frame of predictors, passed down from `parsnip`'s
#'   `data.frame` fit interface (already separated from the outcome).
#' @param y A vector or data frame of outcomes, passed down from `parsnip`'s
#'   `data.frame` fit interface.
#' @param layer_blocks A named list of layer block functions. This is passed
#'   internally from the `parsnip` model specification.
#' @param ... Additional arguments passed down from the model specification.
#'   These can include:
#'   \itemize{
#'     \item \strong{Layer Parameters:} Arguments for the layer blocks, prefixed
#'       with the block name (e.g., `dense_units = 64`).
#'     \item \strong{Architecture Parameters:} Arguments to control the number
#'       of times a block is repeated, in the format `num_{block_name}` (e.g.,
#'       `num_dense = 2`).
#'     \item \strong{Compile Parameters:} Arguments to customize model
#'       compilation, prefixed with `compile_` (e.g., `compile_loss = "mae"`,
#'       `compile_optimizer = "sgd"`).
#'     \item \strong{Fit Parameters:} Arguments to customize model fitting,
#'       prefixed with `fit_` (e.g., `fit_callbacks = list(...)`,
#'       `fit_class_weight = list(...)`).
#'   }
#'
#' @return A list containing the fitted model and other metadata. This list is
#'   stored in the `fit` slot of the `parsnip` model fit object. The list
#'   contains the following elements:
#'   \itemize{
#'     \item `fit`: The raw, fitted Keras model object.
#'     \item `history`: The Keras training history object.
#'     \item `lvl`: A character vector of the outcome factor levels (for
#'       classification) or `NULL` (for regression).
#'   }
#'
#' @examples
#' # This function is not called directly by users.
#' # It is called internally by `parsnip::fit()`.
#' # For example:
#' \donttest{
#' library(parsnip)
#' # create_keras_functional_spec(...) defines my_functional_model
#'
#' # spec <- my_functional_model(hidden_units = 128, fit_epochs = 10) |>
#' #   set_engine("keras")
#'
#' # # This call to fit() would invoke generic_functional_fit() internally
#' # fitted_model <- fit(spec, y ~ x, data = training_data)
#' }
#' @keywords internal
#' @export
generic_functional_fit <- function(
  x,
  y,
  layer_blocks,
  ...
) {
  # --- 1. Build and Compile Model ---
  model <- build_compile_func_model(x, y, layer_blocks, ...)

  # --- 2. Model Fitting ---
  all_args <- list(...)
  verbose <- all_args$verbose %||% 0
  x_processed <- process_x_functional(x)
  x_proc <- x_processed$x_proc
  y_processed <- process_y_functional(y, layer_blocks = layer_blocks)
  y_mat <- y_processed$y_proc

  fit_args <- collect_fit_args(
    x_proc,
    y_mat,
    verbose,
    all_args
  )

  # Fit the model using the constructed arguments
  history <- rlang::exec(keras3::fit, model, !!!fit_args)

  # --- 3. Compute Laplace posterior ---
  if (is_regression_mode(y_processed$class_levels)) {
    laplace <- laplace_all_regression(model, x_proc, y_mat)
  } else {
    laplace <- laplace_all_classification(model, x_proc, y_mat)
  }

  # --- 4. Return value ---
  list(
    fit = model, # The raw Keras model object
    keras_bytes = keras_model_to_bytes(model), # Bytes for RDS-safe restore
    history = history, # The training history
    # Factor levels for classification, NULL for regression
    lvl = y_processed$class_levels,
    # Step/variable structure for a single vector-valued (multi-step
    # regression) output; NULL otherwise. Used by keras_postprocess_numeric().
    multistep_info = y_processed$multistep_info,
    process_x = process_x_functional,
    process_y = process_y_functional,
    laplace = laplace # Laplace posterior (NULL for classification)
  )
}
