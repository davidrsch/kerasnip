#' Remove a Keras Model Specification and its Registrations
#'
#' @description
#' This function completely removes a model specification that was previously
#' created by [create_keras_sequential_spec()] or [create_keras_functional_spec()].
#' It cleans up both the function in the user's environment and all associated
#' registrations within the `parsnip` package.
#'
#' @details
#' This function is essential for cleanly unloading a dynamically created model.
#' It performs three main actions:
#' \enumerate{
#'   \item It removes the model specification function (e.g., `my_mlp()`) and its
#'     corresponding `update()` method from the specified environment.
#'   \item It searches `parsnip`'s internal model environment for all objects
#'     whose names start with the `model_name` and removes them. This purges
#'     the fit methods, argument definitions, and other registrations.
#'   \item It removes the model's name from `parsnip`'s master list of models.
#' }
#' This function uses the un-exported `parsnip:::get_model_env()` to perform
#' the cleanup, which may be subject to change in future `parsnip` versions.
#'
#' @param model_name A character string giving the name of the model
#'   specification function to remove (e.g., "my_mlp").
#' @param env The environment from which to remove the function and its `update()`
#'   method. Defaults to the calling environment (`parent.frame()`).
#' @return Invisibly returns `TRUE` after attempting to remove the objects.
#' @seealso [create_keras_sequential_spec()], [create_keras_functional_spec()]
#' @export
#' @examples
#' \dontrun{
#' if (requireNamespace("keras3", quietly = TRUE)) {
#'   # First, create a dummy spec
#'   input_block <- function(model, input_shape) keras3::keras_model_sequential(input_shape = input_shape)
#'   dense_block <- function(model, units = 16) model |> keras3::layer_dense(units = units)
#'   create_keras_sequential_spec("my_temp_model", list(input = input_block, dense = dense_block), "regression")
#'
#'   # Check it exists in the environment and in parsnip
#'   exists("my_temp_model")
#'   "my_temp_model" %in% parsnip::show_engines("my_temp_model")$model
#'
#'   # Now remove it
#'   remove_keras_spec("my_temp_model")
#'
#'   # Check it's gone
#'   !exists("my_temp_model")
#'   !"my_temp_model" %in% parsnip::show_engines(NULL)$model
#' }
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
