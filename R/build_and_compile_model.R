#' Build and Compile a Keras Sequential Model
#'
#' @description
#' This internal helper function constructs and compiles a Keras sequential model
#' based on a list of layer blocks and other parameters. It handles data
#' processing, dynamic architecture construction, and model compilation.
#'
#' @param x A data frame or matrix of predictors.
#' @param y A vector or data frame of outcomes.
#' @param layer_blocks A named list of functions that define the layers of the
#'   model. The order of the list determines the order of the layers.
#' @param ... Additional arguments passed to the function, including layer
#'   hyperparameters, repetition counts for blocks, and compile/fit arguments.
#'
#' @return A compiled Keras model object.
#' @noRd
build_and_compile_sequential_model <- function(
  x,
  y,
  layer_blocks,
  ...
) {
  # --- 0. Argument & Data Preparation ---
  all_args <- list(...)
  learn_rate <- all_args$learn_rate %||% 0.01
  verbose <- all_args$verbose %||% 0

  # Process x input
  x_processed <- process_x_sequential(x)
  x_proc <- x_processed$x_proc
  input_shape <- x_processed$input_shape

  # Process y input
  y_processed <- process_y_sequential(y)

  # Determine is_classification, class_levels, and num_classes
  is_classification <- y_processed$is_classification
  class_levels <- y_processed$class_levels
  num_classes <- y_processed$num_classes
  y_mat <- y_processed$y_proc

  # Determine default compile arguments based on mode
  default_loss <- if (is_classification) {
    if (num_classes > 2) {
      "categorical_crossentropy"
    } else {
      "binary_crossentropy"
    }
  } else {
    "mean_squared_error"
  }
  default_metrics <- if (is_classification) {
    "accuracy"
  } else {
    "mean_absolute_error"
  }

  # --- 2. Dynamic Model Architecture Construction ---
  # The model is initialized as NULL. The first layer_block is expected to
  # create the model (e.g., by defining an input layer). Subsequent blocks
  # will receive and modify the model object. The order is critical.
  model <- NULL

  for (block_name in names(layer_blocks)) {
    block_fn <- layer_blocks[[block_name]]
    block_fmls <- rlang::fn_fmls(block_fn)

    num_repeats_arg <- paste0("num_", block_name)
    num_repeats_val <- all_args[[num_repeats_arg]]

    # If num_repeats_val is NULL or zapped, default to 1.
    # Otherwise, use the value provided by the user.
    if (is.null(num_repeats_val) || inherits(num_repeats_val, "rlang_zap")) {
      num_repeats <- 1
    } else {
      num_repeats <- as.integer(num_repeats_val)
    }

    # Get the arguments for this specific block from `...`
    block_arg_names <- names(block_fmls)[-1] # Exclude 'model'
    user_args <- list()
    for (arg_name in block_arg_names) {
      full_arg_name <- paste(block_name, arg_name, sep = "_")
      arg_val <- all_args[[full_arg_name]]
      # Only use the argument if it was actually provided by the user
      if (!is.null(arg_val) && !inherits(arg_val, "rlang_zap")) {
        user_args[[arg_name]] <- arg_val
      }
    }

    # Combine user-provided args with the block's defaults
    block_args <- utils::modifyList(as.list(block_fmls[-1]), user_args)

    # If the block function can accept these, provide them. This is useful for
    # the user-defined input and output layers.
    if ("input_shape" %in% names(block_fmls)) {
      block_args$input_shape <- input_shape
    }
    if (is_classification && "num_classes" %in% names(block_fmls)) {
      block_args$num_classes <- num_classes
    }

    # Add the block(s) to the model
    for (i in seq_len(num_repeats)) {
      # The first argument to the block function is the model itself
      # On the first iteration, `model` will be NULL.
      call_args <- c(list(model), block_args)
      model <- rlang::exec(block_fn, !!!call_args)
    }
  }

  # --- 3. Model Compilation ---
  # Collect all arguments starting with "compile_" from `...`
  compile_args <- collect_compile_args(
    all_args,
    learn_rate,
    default_loss,
    default_metrics
  )
  rlang::exec(keras3::compile, model, !!!compile_args)

  return(model)
}

#' Build and Compile a Keras Functional Model
#'
#' @description
#' This internal helper function constructs and compiles a Keras functional model
#' based on a list of layer blocks and other parameters. It handles data
#' processing, dynamic architecture construction (including multiple inputs and
#' branches), and model compilation.
#'
#' @param x A data frame or matrix of predictors. For multiple inputs, this is
#'   often a data frame with list-columns.
#' @param y A vector or data frame of outcomes. Can handle multiple outputs if
#'   provided as a data frame with multiple columns.
#' @param layer_blocks A named list of functions that define the building blocks
#'   of the model graph. Connections are defined by referencing other block names.
#' @param ... Additional arguments passed to the function, including layer
#'   hyperparameters, repetition counts for blocks, and compile/fit arguments.
#'
#' @return A compiled Keras model object.
#' @noRd
build_and_compile_functional_model <- function(
  x,
  y,
  layer_blocks,
  ...
) {
  # --- 0. Argument & Data Preparation ---
  all_args <- list(...)
  learn_rate <- all_args$learn_rate %||% 0.01
  verbose <- all_args$verbose %||% 0

  # Process x input
  x_processed <- process_x_functional(x)
  x_proc <- x_processed$x_proc
  input_shape <- x_processed$input_shape

  # Process y input
  y_processed <- process_y_functional(y)

  # Determine default compile arguments based on mode
  default_losses <- list()
  default_metrics_list <- list()

  # Check if y_processed$y_proc is a list (indicating multiple outputs)
  if (is.list(y_processed$y_proc) && !is.null(names(y_processed$y_proc))) {
    # Multiple outputs
    for (output_name in names(y_processed$y_proc)) {
      # We need to determine is_classification and num_classes for each output
      # based on the class_levels for that output.
      current_class_levels <- y_processed$class_levels[[output_name]]
      current_is_classification <- !is.null(current_class_levels) &&
        length(current_class_levels) > 0
      current_num_classes <- if (current_is_classification) {
        length(current_class_levels)
      } else {
        NULL
      }

      default_losses[[output_name]] <- if (current_is_classification) {
        if (current_num_classes > 2) {
          "categorical_crossentropy"
        } else {
          "binary_crossentropy"
        }
      } else {
        "mean_squared_error"
      }
      default_metrics_list[[output_name]] <- if (current_is_classification) {
        "accuracy"
      } else {
        "mean_absolute_error"
      }
    }
  } else {
    # Single output case
    # Determine is_classification and num_classes from the top-level class_levels
    is_classification <- !is.null(y_processed$class_levels) &&
      length(y_processed$class_levels) > 0
    num_classes <- if (is_classification) {
      length(y_processed$class_levels)
    } else {
      NULL
    }

    default_losses <- if (is_classification) {
      if (num_classes > 2) {
        "categorical_crossentropy"
      } else {
        "binary_crossentropy"
      }
    } else {
      "mean_squared_error"
    }
    default_metrics_list <- if (is_classification) {
      "accuracy"
    } else {
      "mean_absolute_error"
    }
  }

  # --- 2. Dynamic Model Architecture Construction (DIFFERENT from sequential) ---
  # Create a list to store the output tensors of each block.  The names of the
  # list elements correspond to the block names.
  block_outputs <- list()
  model_input_tensors <- list() # To collect all input tensors for keras_model

  # Identify and process input layers based on names matching input_shape
  # This assumes that if input_shape is a named list, the corresponding
  # input blocks in layer_blocks will have matching names.
  if (is.list(input_shape) && !is.null(names(input_shape))) {
    input_block_names_in_spec <- intersect(
      names(layer_blocks),
      names(input_shape)
    )

    if (length(input_block_names_in_spec) != length(input_shape)) {
      stop(
        "Mismatch between named inputs from process_x and named input blocks in layer_blocks. ",
        "Ensure all processed inputs have a corresponding named input block in your model specification."
      )
    }

    for (block_name in input_block_names_in_spec) {
      block_fn <- layer_blocks[[block_name]]
      current_input_tensor <- block_fn(input_shape = input_shape[[block_name]])
      block_outputs[[block_name]] <- current_input_tensor
      model_input_tensors[[block_name]] <- current_input_tensor
    }
    remaining_layer_blocks_names <- names(layer_blocks)[
      !(names(layer_blocks) %in% input_block_names_in_spec)
    ]
  } else {
    # Single input case (original logic, but now also collecting for model_input_tensors)
    first_block_name <- names(layer_blocks)[1]
    first_block_fn <- layer_blocks[[first_block_name]]
    current_input_tensor <- first_block_fn(input_shape = input_shape)
    block_outputs[[first_block_name]] <- current_input_tensor
    model_input_tensors[[first_block_name]] <- current_input_tensor
    remaining_layer_blocks_names <- names(layer_blocks)[-1]
  }

  # Iterate through the remaining blocks, connecting and repeating them as needed.
  for (block_name in remaining_layer_blocks_names) {
    block_fn <- layer_blocks[[block_name]]
    block_fmls <- rlang::fn_fmls(block_fn)
    block_fml_names <- names(block_fmls)

    # --- Get Repetition Count ---
    num_repeats_arg <- paste0("num_", block_name)
    num_repeats_val <- all_args[[num_repeats_arg]]

    # If num_repeats_val is NULL or zapped, default to 1.
    # Otherwise, use the value provided by the user.
    if (is.null(num_repeats_val) || inherits(num_repeats_val, "rlang_zap")) {
      num_repeats <- 1
    } else {
      num_repeats <- as.integer(num_repeats_val)
    }

    # --- Get Hyperparameters for this block ---
    # Hyperparameters are formals that are NOT other block names (graph connections)
    hyperparam_names <- setdiff(block_fml_names, names(layer_blocks))
    user_hyperparams <- list()
    for (hp_name in hyperparam_names) {
      full_arg_name <- paste(block_name, hp_name, sep = "_")
      arg_val <- all_args[[full_arg_name]]
      if (!is.null(arg_val) && !inherits(arg_val, "rlang_zap")) {
        user_hyperparams[[hp_name]] <- arg_val
      }
    }
    # Combine user args with the block's defaults for those hyperparameters
    block_hyperparams <- utils::modifyList(
      as.list(block_fmls[hyperparam_names]),
      user_hyperparams
    )

    # Add special engine-supplied arguments if the block can accept them
    # Add special engine-supplied arguments if the block can accept them
    # This is primarily for output layers that might need num_classes
    if ("num_classes" %in% block_fml_names) {
      # Check if this block is an output block and if it's a classification task
      if (is.list(y_processed$y_proc) && !is.null(names(y_processed$y_proc))) { # Multi-output case
        # Find the corresponding output in y_processed based on block_name
        y_names <- names(y_processed$y_proc)
        # If there is only one output, and this block is named 'output',
        # connect them automatically.
        if (length(y_names) == 1 && block_name == "output") {
          y_name <- y_names[1]
          is_cls <- !is.null(y_processed$class_levels[[y_name]]) &&
            length(y_processed$class_levels[[y_name]]) > 0
          if (is_cls) {
            block_hyperparams$num_classes <- length(y_processed$class_levels[[y_name]])
          }
        } else if (block_name %in% y_names) {
          # Standard case: block name matches an output name
          current_y_info <- list(
            is_classification = !is.null(y_processed$class_levels[[block_name]]) &&
              length(y_processed$class_levels[[block_name]]) > 0,
            num_classes = if (!is.null(y_processed$class_levels[[block_name]])) {
              length(y_processed$class_levels[[block_name]])
            } else {
              NULL
            }
          )
          if (current_y_info$is_classification) {
            block_hyperparams$num_classes <- current_y_info$num_classes
          }
        }
      } else { # Single output case
        if (is_classification) {
          block_hyperparams$num_classes <- num_classes
        }
      }
    }

    # --- Get Input Tensors for this block ---
    input_tensor_names <- intersect(block_fml_names, names(block_outputs))
    if (length(input_tensor_names) == 0 && block_name != "output") {
      warning("Block '", block_name, "' has no inputs from other blocks.")
    }

    # --- Repetition Loop ---
    if (num_repeats > 1 && length(input_tensor_names) != 1) {
      stop(
        "Block '",
        block_name,
        "' cannot be repeated because it has ",
        length(input_tensor_names),
        " inputs (",
        paste(input_tensor_names, collapse = ", "),
        "). Only blocks with exactly one input tensor can be repeated."
      )
    }

    # The initial input(s) for the first iteration
    input_args <- purrr::map(input_tensor_names, ~ block_outputs[[.x]])
    names(input_args) <- input_tensor_names

    # The tensor that will be updated and passed through the loop
    current_tensor <- input_args[[1]]

    for (i in seq_len(num_repeats)) {
      # For repetitions after the first, update the input tensor
      if (i > 1) {
        input_args[[input_tensor_names[1]]] <- current_tensor
      }
      call_args <- c(input_args, block_hyperparams)
      current_tensor <- rlang::exec(block_fn, !!!call_args)
    }

    # Store the final output of the (possibly repeated) block
    block_outputs[[block_name]] <- current_tensor
  }

  # The last layer must be named 'output' or match the names of y_processed outputs
  final_output_tensors <- list()

  # Check if y_processed$y_proc is a named list, indicating multiple outputs)
  if (is.list(y_processed$y_proc) && !is.null(names(y_processed$y_proc))) {
    # Multiple outputs
    for (output_name in names(y_processed$y_proc)) {
      # Iterate over the names of the actual outputs
      if (is.null(block_outputs[[output_name]])) {
        stop(paste0(
          "An output block named '",
          output_name,
          "' must be defined in layer_blocks for multi-output models."
        ))
      }
      final_output_tensors[[output_name]] <- block_outputs[[output_name]]
    }
  } else {
    # Single output case
    output_tensor <- block_outputs[["output"]]
    if (is.null(output_tensor)) {
      stop("An 'output' block must be defined in layer_blocks.")
    }
    final_output_tensors <- output_tensor
  }

  # If there's only one input, it shouldn't be a list for keras_model
  final_model_inputs <- if (length(model_input_tensors) == 1) {
    model_input_tensors[[1]]
  } else {
    model_input_tensors
  }

  model <- keras3::keras_model(
    inputs = final_model_inputs,
    outputs = final_output_tensors # This will now be a list if multiple outputs
  )

  # --- 3. Model Compilation ---
  # Collect all arguments starting with "compile_" from `...`
  compile_args <- collect_compile_args(
    all_args,
    learn_rate,
    default_losses,
    default_metrics_list
  )
  rlang::exec(keras3::compile, model, !!!compile_args)

  return(model)
}
