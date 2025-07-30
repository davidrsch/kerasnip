#' @name keras_objects
#' @title Dynamically Discovered Keras Objects
#' @description
#' These exported vectors contain the names of optimizers, losses, and metrics
#' discovered from the installed `keras3` package when `kerasnip` is loaded.
#' This ensures that `kerasnip` is always up-to-date with your Keras version.
#' @details
#' These objects are primarily used to provide the default `values` for the
#' `dials` parameter functions, [optimizer_function()] and [loss_function_keras()].
#' This allows for tab-completion and validation of optimizer and loss names
#' when tuning models.
#'
#' The discovery process in `.onLoad()` scrapes the `keras3` namespace for
#' functions matching `optimizer_*`, `loss_*`, and `metric_*` patterns.
#' @keywords internal
NULL

#' @rdname keras_objects
#' @export
keras_optimizers <- NULL

#' @rdname keras_objects
#' @export
keras_losses <- NULL

#' @rdname keras_objects
#' @export
keras_metrics <- NULL

#' Populate Keras Object Lists on Package Load
#'
#' @description
#' This `.onLoad` hook is executed when the `kerasnip` package is loaded. Its
#' main purpose is to inspect the installed `keras3` package and populate the
#' `keras_optimizers`, `keras_losses`, and `keras_metrics` vectors.
#'
#' @details
#' The function works by:
#' \enumerate{
#'   \item Checking if `keras3` is installed.
#'   \item Listing all functions in the `keras3` namespace that match the patterns
#'     `optimizer_*`, `loss_*`, and `metric_*`.
#'   \item For each function, it attempts to extract the default value of the `name`
#'     argument (e.g., for `keras3::optimizer_adam()`, it extracts `"adam"`).
#'   \item It populates the exported vectors with these discovered names. For metrics,
#'     it also adds a list of common string aliases that Keras accepts.
#' }
#' This dynamic discovery ensures that `kerasnip` automatically supports all
#' objects available in the user's installed version of Keras.
#' @noRd
.onLoad <- function(libname, pkgname) {
  # Helper to get the default string name from a Keras function's `name` argument
  get_keras_default_name <- function(fn_name, keras_ns) {
    fn <- get(fn_name, envir = keras_ns)
    args <- formals(fn)
    if (!is.null(args$name)) {
      # The default value is a string, e.g., name = "adam"
      return(as.character(args$name))
    }
    NA_character_
  }

  if (!requireNamespace("keras3", quietly = TRUE)) {
    return()
  }
  keras_ns <- asNamespace("keras3")

  opt_fns <- ls(keras_ns, pattern = "^optimizer_")
  loss_fns <- ls(keras_ns, pattern = "^loss_")
  metric_fns <- ls(keras_ns, pattern = "^metric_")

  # Find and assign the vectors to the package's namespace
  assign(
    "keras_optimizers",
    stats::na.omit(purrr::map_chr(opt_fns, get_keras_default_name, keras_ns)),
    envir = parent.env(environment())
  )
  assign(
    "keras_losses",
    stats::na.omit(purrr::map_chr(loss_fns, get_keras_default_name, keras_ns)),
    envir = parent.env(environment())
  )

  # For metrics, we find the metric_* functions and also add common string
  # aliases that Keras accepts, as they don't all have function wrappers.
  discovered_metrics <- stats::na.omit(purrr::map_chr(
    metric_fns,
    get_keras_default_name,
    keras_ns
  ))
  common_metric_aliases <- c(
    "mae",
    "mape",
    "mse",
    "msle",
    "auc",
    "poisson",
    "cosine_similarity"
  )
  all_metrics <- unique(sort(c(discovered_metrics, common_metric_aliases)))

  assign("keras_metrics", all_metrics, envir = parent.env(environment()))
}
