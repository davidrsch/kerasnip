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
  # 1. Remove the spec + update fn from the user env
  if (
    exists(model_name, envir = env, inherits = FALSE) &&
      is.function(get(model_name, envir = env))
  ) {
    remove(list = model_name, envir = env)
  }
  update_fn <- paste0("update.", model_name)
  if (exists(update_fn, envir = env, inherits = FALSE)) {
    remove(list = update_fn, envir = env)
  }

  # 2. Nuke every parsnip object whose name starts with model_name
  model_env <- parsnip:::get_model_env()
  all_regs <- ls(envir = model_env)
  to_kill <- grep(paste0("^", model_name), all_regs, value = TRUE)
  if (length(to_kill)) {
    rm(list = to_kill, envir = model_env)
    message(
      "Removed from parsnip registry objects: ",
      paste(to_kill, collapse = ", ")
    )
  }

  # 3. Remove the entry in get_model_env()$models
  if ("models" %in% all_regs && model_name %in% model_env$models) {
    model_env$models <- model_env$models[-which(model_name == model_env$models)]
    message("Removed '", model_name, "' from parsnip:::get_model_env()$models")
  }

  invisible(TRUE)
}
