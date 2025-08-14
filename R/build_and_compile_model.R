
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
  x_processed <- process_x(x)
  x_proc <- x_processed$x_proc
  input_shape <- x_processed$input_shape

  # Process y input
  y_processed <- process_y(y)
  y_mat <- y_processed$y_proc
  is_classification <- y_processed$is_classification
  class_levels <- y_processed$class_levels
  num_classes <- y_processed$num_classes

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
  compile_args <- collect_compile_args(
    all_args,
    learn_rate,
    default_loss,
    default_metrics
  )
  rlang::exec(keras3::compile, model, !!!compile_args)

  return(model)
}

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
  x_processed <- process_x(x)
  x_proc <- x_processed$x_proc
  input_shape <- x_processed$input_shape

  # Process y input
  y_processed <- process_y(y)
  y_mat <- y_processed$y_proc
  is_classification <- y_processed$is_classification
  class_levels <- y_processed$class_levels
  num_classes <- y_processed$num_classes

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

  # --- 2. Dynamic Model Architecture Construction (DIFFERENT from sequential) ---
  # Create a list to store the output tensors of each block.  The names of the
  # list elements correspond to the block names.
  block_outputs <- list()
  # The first block MUST be the input layer and MUST NOT have `input_from`.
  first_block_name <- names(layer_blocks)[1]
  first_block_fn <- layer_blocks[[first_block_name]]
  block_outputs[[first_block_name]] <- first_block_fn(input_shape = input_shape)

  # Iterate through the remaining blocks, connecting and repeating them as needed.
  for (block_name in names(layer_blocks)[-1]) {
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
    if (is_classification && "num_classes" %in% block_fml_names) {
      block_hyperparams$num_classes <- num_classes
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

  # The last layer must be named 'output'
  output_tensor <- block_outputs[["output"]]
  if (is.null(output_tensor)) {
    stop("An 'output' block must be defined in layer_blocks.")
  }
  model <- keras3::keras_model(
    inputs = block_outputs[[first_block_name]],
    outputs = output_tensor
  )

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
