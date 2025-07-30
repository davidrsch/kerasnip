#' Discover and Collect Model Specification Arguments
#'
#' @description
#' This internal helper introspects the user-provided `layer_blocks` functions
#' to generate a complete list of arguments for the new model specification.
#' The logic for discovering arguments differs for sequential and functional models.
#'
#' @details
#' For **sequential models** (`functional = FALSE`):
#' - It creates `num_{block_name}` arguments to control block repetition.
#' - It inspects the arguments of each block function, skipping the first
#'   (assumed to be the `model` object), to find tunable hyperparameters.
#'
#' For **functional models** (`functional = TRUE`):
#' - It creates `num_{block_name}` arguments to control block repetition.
#' - It inspects the arguments of each block function. Arguments whose names
#'   match other block names are considered graph connections (inputs) and are
#'   ignored. The remaining arguments are treated as tunable hyperparameters.
#'
#' In both cases, it also adds global training parameters (like `epochs`) and
#' filters out special engine-supplied arguments (`input_shape`, `num_classes`).
#'
#' @param layer_blocks A named list of functions defining Keras layer blocks.
#' @param functional A logical. If `TRUE`, uses discovery logic for the
#'   Functional API. If `FALSE`, uses logic for the Sequential API.
#' @param global_args A character vector of global arguments to add to the
#'   specification (e.g., "epochs").
#' @return A list containing two elements:
#'
#' @noRd
collect_spec_args <- function(
  layer_blocks,
  functional
) {
  if (any(c("compile", "fit", "optimizer") %in% names(layer_blocks))) {
    stop(
      "`compile`, `fit` and `optimizer` are protected names and cannot be used as layer block names.",
      call. = FALSE
    )
  }

  all_args <- list()
  parsnip_names <- character()

  block_names <- names(layer_blocks)

  # block repetition counts (e.g., num_dense)
  for (block_name in block_names) {
    num_name <- paste0("num_", block_name)
    all_args[[num_name]] <- rlang::zap()
    parsnip_names <- c(parsnip_names, num_name)
  }

  # These args are passed by the fit engine, not set by the user in the spec
  engine_args <- c("input_shape", "num_classes")
  # Discover block-specific hyperparameters
  for (block_name in block_names) {
    block_fmls <- rlang::fn_fmls(layer_blocks[[block_name]])

    if (isTRUE(functional)) {
      # For functional models, hyperparameters are arguments that are NOT
      # names of other blocks (which are graph connections).
      hyperparam_names <- setdiff(
        names(block_fmls),
        c(block_names, engine_args)
      )
    } else {
      # For sequential models, hyperparameters are all args except the first
      # ('model') and special engine args.
      fmls_to_process <- if (length(block_fmls) > 0) block_fmls[-1] else list()
      hyperparam_names <- names(fmls_to_process)[
        !names(fmls_to_process) %in% engine_args
      ]
    }

    for (arg in hyperparam_names) {
      full <- paste0(block_name, "_", arg)
      all_args[[full]] <- rlang::zap()
      parsnip_names <- c(parsnip_names, full)
    }
  }

  # Add global training and compile parameters dynamically
  # These are discovered from keras3::fit and keras3::compile in zzz.R
  fit_params <- if (length(keras_fit_arg_names) > 0) {
    paste0("fit_", keras_fit_arg_names)
  } else {
    character()
  }
  compile_params <- if (length(keras_compile_arg_names) > 0) {
    paste0("compile_", keras_compile_arg_names)
  } else {
    character()
  }

  # learn_rate is a special convenience argument for the default optimizer
  special_params <- "learn_rate"

  dynamic_global_args <- c(special_params, fit_params, compile_params)

  for (g in dynamic_global_args) {
    all_args[[g]] <- rlang::zap()
    parsnip_names <- c(parsnip_names, g)
  }

  list(all_args = all_args, parsnip_names = parsnip_names)
}

#' Internal Implementation for Creating Keras Specifications
#'
#' @description
#' This is the core implementation for both `create_keras_sequential_spec()` and
#' `create_keras_functional_spec()`. It orchestrates the argument collection,
#' function building, and `parsnip` registration steps.
#'
#' @inheritParams create_keras_sequential_spec
#' @param functional A logical, if `TRUE`, registers the model to be fit with
#'   the Functional API (`generic_functional_fit`). Otherwise, uses the
#'   Sequential API (`generic_sequential_fit`).
#'
#' @noRd
create_keras_spec_impl <- function(
  model_name,
  layer_blocks,
  mode,
  functional,
  env
) {
  args_info <- collect_spec_args(layer_blocks, functional = functional)
  spec_fun <- build_spec_function(
    model_name,
    mode,
    args_info$all_args,
    args_info$parsnip_names,
    layer_blocks,
    functional = functional
  )

  register_core_model(model_name, mode)
  register_model_args(model_name, args_info$parsnip_names)
  register_fit_predict(model_name, mode, layer_blocks, functional = functional)
  register_update_method(model_name, args_info$parsnip_names, env = env)

  rlang::env_poke(env, model_name, spec_fun)
  invisible(NULL)
}
