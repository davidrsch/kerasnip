#' @name keras_objects
#' @title Dynamically Discovered Keras Objects
#' @description
#' These vectors contain the names of optimizers, losses, and metrics
#' discovered from the installed `keras3` package at load time. This ensures
#' that `kerasnip` is always up-to-date with your Keras version.
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
