#' Remove a Keras Model Specification
#'
#' This function removes a model specification function that was previously
#' created by `create_keras_spec()` from an environment.
#'
#' @param model_name A character string giving the name of the model
#'   specification function to remove.
#' @param env The environment from which to remove the function. Defaults to
#'   the calling environment (`parent.frame()`), which is typically where
#'   `create_keras_spec()` would have created the function.
#' @return Invisibly returns `TRUE` if the function was found and removed,
#'   and `FALSE` otherwise.
#' @export
#' @examples
#' \dontrun{
#' # First, create a dummy spec
#' dense_block <- function(model, units = 16) {
#'   model |> keras3::layer_dense(units = units)
#' }
#' create_keras_spec("my_temp_model", list(dense = dense_block), "regression")
#'
#' # Check it exists
#' exists("my_temp_model")
#'
#' # Now remove it
#' remove_keras_spec("my_temp_model")
#'
#' # Check it's gone
#' !exists("my_temp_model")
#' }
remove_keras_spec <- function(model_name, env = parent.frame()) {
  spec_found <- FALSE
  if (exists(model_name, envir = env, inherits = FALSE)) {
    obj <- get(model_name, envir = env)
    if (is.function(obj)) {
      remove(list = model_name, envir = env)
      spec_found <- TRUE
    }
  }

  # Also remove the associated update method
  update_method_name <- paste0("update.", model_name)
  # The update method is in the package namespace. `environment()` inside a
  # package function returns the package namespace.
  pkg_env <- environment()
  if (exists(update_method_name, envir = pkg_env, inherits = FALSE)) {
    remove(list = update_method_name, envir = pkg_env)
  }

  invisible(spec_found)
}
