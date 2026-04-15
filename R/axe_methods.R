#' Butcher axe methods for kerasnip_model_fit
#'
#' @description
#' These methods allow `butcher::butcher()` to reduce the memory footprint of
#' fitted kerasnip model objects. The Keras model itself (stored as raw bytes
#' in `$fit$keras_bytes`) is always preserved so that `predict()` continues
#' to work after butchering.
#'
#' The main saving comes from `axe_data()`, which removes the training history
#' object (`$fit$history`). For long training runs this can be several MB.
#'
#' @param x A `kerasnip_model_fit` object.
#' @param verbose Logical. Print information about memory released and
#'   disabled functions. Default is `FALSE`.
#' @param ... Not used.
#' @return An axed `kerasnip_model_fit` object with the `butcher_kerasnip_model_fit`
#'   class prepended.
#' @name axe-kerasnip_model_fit
#' @keywords internal
NULL

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_data
axe_data.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  old <- x
  x$fit$history <- NULL
  butcher:::add_butcher_attributes(
    x,
    old,
    verbose = verbose,
    disabled = c("extract_keras_history")
  )
}

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_env
axe_env.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  # Intentional no-op: Keras R6 objects rely on Python environments.
  # Stripping R environments from them is unsafe and would break predict().
  butcher:::add_butcher_attributes(x, x, verbose = verbose)
}

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_call
axe_call.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  # No-op: kerasnip fit objects do not store a call component.
  butcher:::add_butcher_attributes(x, x, verbose = verbose)
}

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_ctrl
axe_ctrl.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  # No-op: kerasnip fit objects do not store training controls.
  butcher:::add_butcher_attributes(x, x, verbose = verbose)
}

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_fitted
axe_fitted.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  # No-op: kerasnip does not store fitted values separately from the model.
  butcher:::add_butcher_attributes(x, x, verbose = verbose)
}
