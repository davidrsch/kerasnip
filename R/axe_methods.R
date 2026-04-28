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
NULL

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_data
axe_data.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  old <- x
  x$fit$history <- NULL
  add_butcher_attributes(
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
  add_butcher_attributes(x, x, verbose = verbose)
}

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_call
axe_call.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  # No-op: kerasnip fit objects do not store a call component.
  add_butcher_attributes(x, x, verbose = verbose)
}

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_ctrl
axe_ctrl.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  # No-op: kerasnip fit objects do not store training controls.
  add_butcher_attributes(x, x, verbose = verbose)
}

#' @rdname axe-kerasnip_model_fit
#' @exportS3Method butcher::axe_fitted
axe_fitted.kerasnip_model_fit <- function(x, verbose = FALSE, ...) {
  # No-op: kerasnip does not store fitted values separately from the model.
  add_butcher_attributes(x, x, verbose = verbose)
}

# Copied from tidymodels/butcher R/ui.R (internal, not exported by that package).
# Using ::: on unexported functions is disallowed by CRAN policy, so these
# helpers are reproduced here under butcher's MIT licence.
#' @importFrom lobstr obj_size
#' @importFrom cli cli_alert_info cli_alert_success cli_alert_danger

get_object_size <- function(x, attempts = 5) {
  for (i in seq_len(attempts)) {
    res <- try(lobstr::obj_size(x), silent = TRUE)
    if (!inherits(res, "try-error")) {
      break()
    }
  }
  if (inherits(res, "try-error")) {
    cli::cli_inform(
      "{.fn lobstr::obj_size} failed after {attempts} attempts. Falling back on {.fn object.size}."
    )
    res <- utils::object.size(x)
  }

  res
}

#' Console Messages
#'
#' These console messages are created such that the user is
#' aware of the effects of removing specific components from
#' the model object.
#'
#' @param og Original model object.
#' @param butchered Butchered model object.
#'
#' @keywords internal
#' @name ui

#' @noRd
memory_released <- function(og, butchered) {
  old <- get_object_size(og)
  new <- get_object_size(butchered)
  rel <- old - new
  if (length(rel) == 1) {
    if (isTRUE(all.equal(old, new))) {
      return(NULL)
    }
    return(rel)
  } else {
    return(NULL)
  }
}

#' @noRd
assess_object <- function(og, butchered) {
  mem <- memory_released(og, butchered)
  disabled <- attr(butchered, "butcher_disabled")
  class_added <- grep("butcher", class(butchered)[1])
  if (is.null(mem)) {
    cli::cli_alert_danger("No memory released. Do not butcher.")
  } else {
    abs_mem <- format(abs(mem), big.mark = ",", scientific = FALSE)
    if (mem < 0) {
      cli::cli_alert_danger(
        "The butchered object is {.field {abs_mem}} larger than the original. Do not butcher."
      )
    } else {
      cli::cli_alert_success("Memory released: {.field {abs_mem}}")
      if (!is.null(disabled)) {
        cli::cli_alert_danger("Disabled: {.code {disabled}}")
      }
      if (length(class_added) == 0) {
        class_name <- "butchered"
        cli::cli_alert_danger("Could not add {.cls {class_name}} class")
      }
    }
  }
}

# butcher attributes helper
add_butcher_disabled <- function(x, disabled = NULL) {
  current <- attr(x, "butcher_disabled")
  if (!is.null(disabled)) {
    disabled <- union(current, disabled)
    attr(x, "butcher_disabled") <- disabled
  }
  x
}

# class assignment helper
add_butcher_class <- function(x) {
  if (!any(grepl("butcher", class(x)))) {
    class(x) <- append(paste0("butchered_", class(x)[1]), class(x))
  }
  x
}

# butcher attributes wrapper
add_butcher_attributes <- function(
  x,
  old,
  disabled = NULL,
  add_class = TRUE,
  verbose = FALSE
) {
  if (!identical(x, old)) {
    x <- add_butcher_disabled(x, disabled)
    if (add_class) {
      x <- add_butcher_class(x)
    }
  }
  if (verbose & !missing(old)) {
    assess_object(old, x)
  }
  x
}
