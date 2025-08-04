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

#' Remap Layer Block Arguments for Model Specification
#'
#' @description
#' Creates a wrapper function around a Keras layer block to rename its
#' arguments. This is a powerful helper for defining the `layer_blocks` in
#' [create_keras_functional_spec()] and [create_keras_sequential_spec()],
#' allowing you to connect reusable blocks into a model graph without writing
#' verbose anonymous functions.
#'
#' @details
#' `inp_spec()` makes your model definitions cleaner and more readable. It
#' handles the metaprogramming required to create a new function with the
#' correct argument names, while preserving the original block's hyperparameters
#' and their default values.
#'
#' The function supports two modes of operation based on `input_map`:
#' 1.  **Single Input Renaming**: If `input_map` is a single character string,
#'     the wrapper function renames the *first* argument of the `block` function
#'     to the provided string. This is the common case for blocks that take a
#'     single tensor input.
#' 2.  **Multiple Input Mapping**: If `input_map` is a named character vector,
#'     it provides an explicit mapping from new argument names (the names of the
#'     vector) to the original argument names in the `block` function (the values
#'     of the vector). This is used for blocks with multiple inputs, like a
#'     concatenation layer.
#'
#' @param block A function that defines a Keras layer or a set of layers. The
#'   first arguments should be the input tensor(s).
#' @param input_map A single character string or a named character vector that
#'   specifies how to rename/remap the arguments of `block`.
#'
#' @return A new function (a closure) that wraps the `block` function with
#'   renamed arguments, ready to be used in a `layer_blocks` list.
#'
#' @export
#' @examples
#' \dontrun{
#' # --- Example Blocks ---
#' # A standard dense block with one input tensor and one hyperparameter.
#' dense_block <- function(tensor, units = 16) {
#'   tensor |> keras3::layer_dense(units = units, activation = "relu")
#' }
#'
#' # A block that takes two tensors as input.
#' concat_block <- function(input_a, input_b) {
#'   keras3::layer_concatenate(list(input_a, input_b))
#' }
#'
#' # An output block with one input.
#' output_block <- function(tensor) {
#'   tensor |> keras3::layer_dense(units = 1)
#' }
#'
#' # --- Usage ---
#' layer_blocks <- list(
#'   main_input = keras3::layer_input,
#'   path_a = inp_spec(dense_block, "main_input"),
#'   path_b = inp_spec(dense_block, "main_input"),
#'   concatenated = inp_spec(
#'     concat_block,
#'     c(path_a = "input_a", path_b = "input_b")
#'   ),
#'   output = inp_spec(output_block, "concatenated")
#' )
#' }
inp_spec <- function(block, input_map) {
  new_fun <- function() {}
  original_formals <- formals(block)
  original_names <- names(original_formals)

  if (length(original_formals) == 0) {
    stop("The 'block' function must have at least one argument.")
  }

  new_formals <- original_formals

  if (
    is.character(input_map) &&
      is.null(names(input_map)) &&
      length(input_map) == 1
  ) {
    # Case 1: Single string, rename first argument
    names(new_formals)[1] <- input_map
  } else if (is.character(input_map) && !is.null(names(input_map))) {
    # Case 2: Named vector for mapping
    if (!all(input_map %in% original_names)) {
      missing_args <- input_map[!input_map %in% original_names]
      stop(paste(
        "Argument(s)",
        paste(shQuote(missing_args), collapse = ", "),
        "not found in the block function."
      ))
    }
    # Use match() for a more concise, vectorized replacement of names
    new_names <- original_names
    match_indices <- match(input_map, original_names)
    new_names[match_indices] <- names(input_map)
    names(new_formals) <- new_names
  } else {
    stop("`input_map` must be a single string or a named character vector.")
  }

  formals(new_fun) <- new_formals

  call_args <- lapply(names(new_formals), as.symbol)
  names(call_args) <- original_names

  body(new_fun) <- as.call(c(list(as.symbol("block")), call_args))
  environment(new_fun) <- environment()
  new_fun
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
