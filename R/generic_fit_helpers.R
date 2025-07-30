#' Collect and Finalize Compilation Arguments
#'
#' @description
#' This internal helper extracts all arguments prefixed with `compile_` from a
#' list of arguments, resolves them, and combines them with defaults.
#'
#' @details
#' It handles the special logic for the `optimizer`, where a string name is
#' resolved to a Keras optimizer object, applying the `learn_rate` if necessary.
#' It also resolves string names for `loss` and `metrics` using `get_keras_object()`.
#'
#' @param all_args The list of all arguments passed to the fitting function's `...`.
#' @param learn_rate The main `learn_rate` parameter.
#' @param default_loss The default loss function to use if not provided.
#' @param default_metrics The default metric(s) to use if not provided.
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

  # Resolve loss: use user-provided, otherwise default. Resolve string if needed.
  loss_arg <- user_compile_args$loss %||% default_loss
  if (is.character(loss_arg)) {
    final_compile_args$loss <- get_keras_object(loss_arg, "loss")
  } else {
    final_compile_args$loss <- loss_arg
  }

  # Resolve metrics: user‐supplied or default
  metrics_arg <- user_compile_args$metrics %||% default_metrics
  # Keras' `compile()` can handle a single string or a list/vector of strings.
  # This correctly passes along either the default string or a user-provided vector.
  final_compile_args$metrics <- metrics_arg

  # Add any other user-provided compile arguments (e.g., `weighted_metrics`)
  other_args <- user_compile_args[
    !names(user_compile_args) %in% c("optimizer", "loss", "metrics")
  ]
  final_compile_args <- c(final_compile_args, other_args)
  final_compile_args
}

#' Collect and Finalize Fitting Arguments
#'
#' @description
#' This internal helper extracts all arguments prefixed with `fit_` from a list
#' of arguments and combines them with the core arguments for `keras3::fit()`.
#'
#' @param x_proc The processed predictor data.
#' @param y_mat The processed outcome data.
#' @param epochs The number of epochs.
#' @param batch_size The batch size.
#' @param validation_split The validation split proportion.
#' @param verbose The verbosity level.
#' @param all_args The list of all arguments passed to the fitting function's `...`.
#' @return A named list of arguments ready to be passed to `keras3::fit()`.
#' @noRd
collect_fit_args <- function(
  x_proc,
  y_mat,
  epochs,
  batch_size,
  validation_split,
  verbose,
  all_args
) {
  # Collect all arguments starting with "fit_" from `...`
  fit_arg_names <- names(all_args)[startsWith(names(all_args), "fit_")]
  user_fit_args <- all_args[fit_arg_names]
  names(user_fit_args) <- sub("^fit_", "", names(user_fit_args))

  final_fit_args <- c(
    list(
      x = x_proc,
      y = y_mat,
      epochs = epochs,
      batch_size = batch_size,
      validation_split = validation_split,
      verbose = verbose
    ),
    user_fit_args
  )
}
