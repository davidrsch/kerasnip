#' Generic Fitting Function for Sequential Keras Models
#'
#' @title Internal Fitting Engine for Sequential API Models
#' @description
#' This function serves as the internal engine for fitting `kerasnip` models
#' that are based on the Keras sequential API. It is not intended to be called
#' directly by the user. The function is invoked by `parsnip::fit()` when a
#' `kerasnip` sequential model specification is used.
#'
#' @details
#' The function orchestrates the three main steps of the model fitting process:
#' \enumerate{
#'   \item \strong{Build and Compile:} It calls
#'     `build_compile_seq_model()` to construct the Keras model
#'     architecture based on the provided `layer_blocks` and hyperparameters.
#'   \item \strong{Process Data:} It preprocesses the input (`x`) and output
#'     (`y`) data into the format expected by Keras.
#'   \item \strong{Fit Model:} It calls `keras3::fit()` with the compiled model
#'     and processed data, passing along any fitting-specific arguments (e.g.,
#'     `epochs`, `batch_size`, `callbacks`).
#' }
#'
#' @section The `fit_seed` argument and `int_conformal_full`:
#' `fit_seed` sets `keras3::set_random_seed()` at the start of every training
#' run. For ordinary use it is optional.
#'
#' `probably::int_conformal_full()` refits the model from scratch for every
#' candidate trial value on every new test observation. Its bounds-searching
#' algorithm assumes that, as the trial outcome sweeps across the grid, the
#' nonconformity score for the trial row changes **monotonically**. This holds
#' for deterministic models (linear regression, random forests, etc.), but a
#' neural network re-trained from random weight initialisation on each refit
#' produces a different function every time. The nonconformity scores across the
#' grid may therefore be non-monotone, in which case `probably` cannot find the
#' interval boundaries and returns `NA`. Whether this actually occurs depends on
#' the data, architecture, and random state of the session — it is not
#' guaranteed on every run, which makes it particularly hard to detect.
#'
#' Setting `fit_seed` makes each refit deterministic and removes this source of
#' unreliability. Without it, kerasnip emits a warning when `int_conformal_full`
#' is detected in the call stack, because the risk is always present even when
#' the current run happens to succeed. Use `int_conformal_split` or
#' `int_conformal_cv` if you want to avoid this constraint entirely, they
#' calibrate a single fixed model and are not affected.
#'
#' @param formula A formula specifying the predictor and outcome variables,
#'   passed down from the `parsnip::fit()` call.
#' @param data A data frame containing the training data, passed down from the
#'   `parsnip::fit()` call.
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
#'     \item \strong{`fit_seed`}: An optional integer passed to
#'       `keras3::set_random_seed()` before every training run. Required for
#'       reliable use with `probably::int_conformal_full()`. See the `fit_seed`
#'       section above.
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
#' # create_keras_sequential_spec(...) defines my_sequential_model
#'
#' # spec <- my_sequential_model(hidden_1_units = 128, fit_epochs = 10) |>
#' #   set_engine("keras")
#'
#' # # This call to fit() would invoke generic_sequential_fit() internally
#' # fitted_model <- fit(spec, y ~ x, data = training_data)
#' }
#' @keywords internal
#' @export
generic_sequential_fit <- function(
  formula,
  data,
  layer_blocks,
  ...
) {
  # Separate predictors and outcomes from the processed data frame provided by
  # parsnip
  y_names <- all.vars(formula[[2]])
  x_names <- all.vars(formula[[3]])

  # Handle the `.` case for predictors
  if ("." %in% x_names) {
    x <- data[, !(names(data) %in% y_names), drop = FALSE]
  } else {
    x <- data[, x_names, drop = FALSE]
  }
  y <- data[, y_names, drop = FALSE]

  # --- 0. Handle fit_seed ---
  # fit_seed is consumed here and stripped from all_args before forwarding to
  # build_compile_seq_model / collect_fit_args so those functions never receive
  # an unexpected argument.
  #
  # When fit_seed is absent and the call is happening inside
  # predict.int_conformal_full we warn unconditionally. The risk is always
  # present: each refit starts from a different random initialisation, which
  # may produce non-monotone nonconformity scores and NA intervals. Whether
  # NA actually occurs depends on the data and session state, it is not
  # guaranteed on every run, which is precisely what makes it dangerous. The
  # warning fires reliably so the user is always informed, even on runs that
  # happen to succeed.
  all_args <- list(...)
  fit_seed <- all_args$fit_seed %||% NULL
  all_args$fit_seed <- NULL

  if (is.null(fit_seed) && .is_inside_conformal_full()) {
    warning(
      "int_conformal_full() refits the model from scratch for every trial ",
      "value. Without `fit_seed`, each refit starts from a different random ",
      "initialisation, so nonconformity scores across the trial grid may be ",
      "non-monotone. When that happens, probably cannot find the interval ",
      "boundaries and returns NA — but the problem does not occur on every run, ",
      "so results that look valid may not be reproducible or trustworthy.\n",
      "Fix: add `fit_seed = <integer>` to your model spec, e.g.:\n",
      "  my_model(fit_epochs = 30, fit_seed = 42L) |> set_engine(\"keras\")\n",
      "Alternative: use int_conformal_split() or int_conformal_cv() instead, ",
      "which calibrate a single fixed model and are not affected by this issue.",
      call. = FALSE
    )
  }

  if (!is.null(fit_seed)) {
    keras3::set_random_seed(as.integer(fit_seed))
  }

  # --- 1. Build and Compile Model ---
  # Forward all_args (fit_seed already removed) via rlang::exec so downstream
  # functions do not receive an unknown argument.
  model <- rlang::exec(build_compile_seq_model, x, y, layer_blocks, !!!all_args)

  # --- 2. Model Fitting ---
  verbose <- all_args$verbose %||% 0
  x_processed <- process_x_sequential(x)
  x_proc <- x_processed$x_proc
  y_processed <- process_y_sequential(y)
  y_mat <- y_processed$y_proc

  fit_args <- collect_fit_args(
    x_proc,
    y_mat,
    verbose,
    all_args
  )

  # Fit the model using the constructed arguments
  history <- rlang::exec(keras3::fit, model, !!!fit_args)

  # --- 3. Return value ---
  list(
    fit = model, # The raw Keras model object
    keras_bytes = keras_model_to_bytes(model), # Bytes for RDS-safe restore
    history = history, # The training history
    # Factor levels for classification, NULL for regression
    lvl = y_processed$class_levels,
    process_x = process_x_sequential,
    process_y = process_y_sequential
  )
}

# Internal helper ---------------------------------------------------------------
# Returns TRUE when the current call stack contains a call to
# predict.int_conformal_full or its known internal helpers. This lets us issue
# the warning at the point where it is actionable (inside the refit loop) rather
# than asking users to anticipate the problem before they run predict().
#
# The function names checked are the public S3 method plus the private helpers
# used by probably 1.x. If probably renames its internals in a future release
# the public method name alone is sufficient to catch the case.
.is_inside_conformal_full <- function() {
  calls <- sys.calls()
  target_fns <- c(
    "predict.int_conformal_full",
    "conformal_full_one_row", # probably internal (1.x)
    "full_conformal_search" # probably internal (alternative name)
  )
  any(vapply(
    calls,
    function(cl) {
      fn <- cl[[1L]]
      is.name(fn) && as.character(fn) %in% target_fns
    },
    logical(1L)
  ))
}
