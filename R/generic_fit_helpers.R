#' Collect and Finalize Compilation Arguments
#'
#' @description
#' This internal helper extracts all arguments prefixed with `compile_` from a
#' list of arguments, resolves them, and combines them with defaults.
#'
#' @details
#' This function orchestrates the compilation setup. It gives precedence to
#' user-provided arguments (e.g., `compile_optimizer`) over the mode-based
#' defaults. It handles the special logic for the `optimizer`, where a string
#' name (e.g., `"sgd"`) is resolved to a Keras optimizer object, applying the
#' top-level `learn_rate` if necessary. It also resolves string names for `loss`
#' and `metrics` using `get_keras_object()`.
#'
#' @param all_args The list of all arguments passed to the fitting function's `...`.
#' @param learn_rate The top-level `learn_rate` parameter.
#' @param default_loss The default loss function to use if not provided. Can be a single value or a named list.
#' @param default_metrics The default metric(s) to use if not provided. Can be a single value or a named list of vectors/single values.
#' @return A named list of arguments ready to be passed to `keras3::compile()`.
#' @noRd
collect_compile_args <- function(
  all_args,
  learn_rate,
  default_loss,
  default_metrics
) {
  compile_arg_names <- names(all_args)[startsWith(names(all_args), "compile_")]
  user_compile_args <- all_args[compile_arg_names]
  names(user_compile_args) <- sub("^compile_", "", names(user_compile_args))

  # --- 3a. Resolve and Finalize Compile Arguments ---
  final_compile_args <- list()

  # Determine the final optimizer object, ensuring `learn_rate` is applied.
  optimizer_arg <- user_compile_args$optimizer %||% NULL
  if (!is.null(optimizer_arg)) {
    if (is.character(optimizer_arg)) {
      # Resolve string to object, passing the learn_rate
      final_compile_args$optimizer <- get_keras_object(
        optimizer_arg,
        "optimizer",
        learning_rate = learn_rate
      )
    } else {
      # User passed a pre-constructed optimizer object, use it as is.
      # We assume they have configured the learning rate within it.
      final_compile_args$optimizer <- optimizer_arg
    }
  } else {
    # No optimizer provided, use the default (Adam) with the learn_rate.
    final_compile_args$optimizer <- keras3::optimizer_adam(
      learning_rate = learn_rate
    )
  }

  # Handle loss: can be single or multiple outputs
  if (is.list(default_loss) && !is.null(names(default_loss))) {
    # Multiple outputs
    # User can provide a single loss for all outputs, or a named list
    loss_arg <- user_compile_args$loss %||% default_loss
    if (is.character(loss_arg) && length(loss_arg) == 1) {
      # Single loss string for all outputs
      final_compile_args$loss <- get_keras_object(loss_arg, "loss")
    } else if (is.list(loss_arg) && !is.null(names(loss_arg))) {
      # Named list of losses
      final_compile_args$loss <- lapply(loss_arg, function(l) {
        if (is.character(l)) get_keras_object(l, "loss") else l
      })
    } else {
      stop(
        "For multiple outputs, 'compile_loss' must be a single string or a named list of losses."
      )
    }
  } else { # Single output
    loss_arg <- user_compile_args$loss %||% default_loss
    if (is.character(loss_arg)) {
      final_compile_args$loss <- get_keras_object(loss_arg, "loss")
    } else {
      final_compile_args$loss <- loss_arg
    }
  }

  # Handle metrics: can be single or multiple outputs
  if (is.list(default_metrics) && !is.null(names(default_metrics))) { # Multiple outputs
    # User can provide a single metric for all outputs, or a named list
    metrics_arg <- user_compile_args$metrics %||% default_metrics
    if (is.character(metrics_arg) && length(metrics_arg) == 1) { # Single metric string for all outputs
      final_compile_args$metrics <- get_keras_object(metrics_arg, "metric")
    } else if (is.list(metrics_arg) && !is.null(names(metrics_arg))) { # Named list of metrics
      final_compile_args$metrics <- lapply(metrics_arg, function(m) {
        if (is.character(m)) get_keras_object(m, "metric") else m
      })
    } else {
      stop("For multiple outputs, 'compile_metrics' must be a single string or a named list of metrics.")
    }
  } else { # Single output
    metrics_arg <- user_compile_args$metrics %||% default_metrics
    if (is.character(metrics_arg)) {
      final_compile_args$metrics <- lapply(metrics_arg, get_keras_object, "metric")
    } else {
      final_compile_args$metrics <- metrics_arg
    }
  }

  # Add any other user-provided compile arguments (e.g., `weighted_metrics`)
  other_args <- user_compile_args[
    !names(user_compile_args) %in% c("optimizer", "loss", "metrics")
  ]
  final_compile_args <- c(final_compile_args, other_args)
  # Filter out arguments that are NULL or rlang_zap before passing to keras3::compile
  final_compile_args <- final_compile_args[
    !vapply(
      final_compile_args,
      function(x) inherits(x, "rlang_zap"),
      logical(1)
    )
  ]
  final_compile_args
}

#' Collect and Finalize Fitting Arguments
#'
#' @description
#' This internal helper extracts all arguments prefixed with `fit_` from a list
#' of arguments and combines them with the core arguments for `keras3::fit()`.
#'
#' @details
#' It constructs the final list of arguments for `keras3::fit()`. It starts with
#' the required data (`x`, `y`) and the `verbose` setting. It then merges any
#' user-provided arguments from the model specification (e.g., `fit_epochs`,
#' `fit_callbacks`), with the user-provided arguments taking precedence over
#' any defaults.
#'
#' @param x_proc The processed predictor data.
#' @param y_mat The processed outcome data.
#' @param verbose The verbosity level.
#' @param all_args The list of all arguments passed to the fitting function's `...`.
#' @return A named list of arguments ready to be passed to `keras3::fit()`.
#' @noRd
collect_fit_args <- function(
  x_proc,
  y_mat,
  verbose,
  all_args
) {
  # Collect all arguments starting with "fit_" from `...`
  fit_arg_names <- names(all_args)[startsWith(names(all_args), "fit_")]
  user_fit_args <- all_args[fit_arg_names]
  names(user_fit_args) <- sub("^fit_", "", names(user_fit_args))

  # Build the core argument set. `verbose` can be overridden by `fit_verbose`.
  base_args <- list(
    x = x_proc,
    y = y_mat,
    verbose = verbose
  )

  merged_args <- utils::modifyList(base_args, user_fit_args)

  # Filter out arguments that are NULL or rlang_zap before passing to keras3::fit
  merged_args <- merged_args[
    !vapply(
      merged_args,
      function(x) {
        inherits(x, "rlang_zap")
      },
      logical(1)
    )
  ]
  merged_args
}