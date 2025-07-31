#' Register the `update()` S3 Method
#'
#' @description
#' Creates and registers an `update()` S3 method for the new model specification.
#' This method is essential for tuning with `dials` and `tune`, as it allows
#' the tuning machinery to modify model parameters after the spec has been created.
#'
#' @details
#' This function uses `rlang` metaprogramming to dynamically construct a complete
#' `update.{{model_name}}` function. The process involves:
#' \enumerate{
#'   \item Building a function signature that includes `object`, `parameters`,
#'     `...`, `fresh`, and all the tunable parameters from `parsnip_names`.
#'   \item Creating a function body that captures all the arguments into quosures
#'     and passes them to `parsnip::update_spec()`.
#'   \item Registering this new function as an S3 method for the generic
#'     `update()` in the specified environment, so S3 dispatch can find it.
#' }
#'
#' @param model_name The name of the new model specification (e.g., "my_mlp").
#' @param parsnip_names A character vector of all argument names that the
#'   `update()` method should be able to modify.
#' @param env The environment in which to create the `update()` S3 method.
#' @return Invisibly returns `NULL`. Called for its side effects.
#' @noRd
register_update_method <- function(model_name, parsnip_names, env) {
  # Build function signature
  update_args_list <- c(
    list(object = rlang::missing_arg(), parameters = rlang::expr(NULL)),
    purrr::map(parsnip_names, ~ rlang::expr(NULL)),
    list(... = rlang::missing_arg(), fresh = rlang::expr(FALSE))
  )
  names(update_args_list)[3:(2 + length(parsnip_names))] <- parsnip_names

  # Create a list of expressions like `arg_name = rlang::enquo(arg_name)`
  args_enquo_exprs <- purrr::map(
    parsnip_names,
    ~ rlang::expr(rlang::enquo(!!rlang::sym(.x)))
  )
  names(args_enquo_exprs) <- parsnip_names

  # Create the expression that builds this list inside the function body
  args_enquo_list_expr <- rlang::expr(
    args <- rlang::list2(!!!args_enquo_exprs)
  )

  # Create the call to `parsnip::update_spec`
  update_spec_call <- rlang::expr(
    parsnip::update_spec(
      object = object,
      parameters = parameters,
      args_enquo_list = args,
      fresh = fresh,
      cls = !!model_name,
      ...
    )
  )

  # Combine them into the final body
  update_body <- rlang::call2("{", args_enquo_list_expr, update_spec_call)

  # Create and register the S3 method
  update_func <- rlang::new_function(
    args = update_args_list,
    body = update_body
  )
  method_name <- paste0("update.", model_name)
  # Poke the function into the target environment (e.g., .GlobalEnv) so that
  # S3 dispatch can find it.
  rlang::env_poke(env, method_name, update_func)
  registerS3method("update", model_name, update_func, envir = env)
}
