#' Discover and Collect Model Specification Arguments
#'
#' Introspects the provided layer block functions to generate a list of
#' arguments for the new model specification. This includes arguments for
#' block repetition (`num_*`), block-specific hyperparameters (`block_*`),
#' and global training parameters.
#'
#' @param layer_blocks A named list of functions defining Keras layer blocks.
#' @param global_args A character vector of global arguments to add to the
#'   specification (e.g., "epochs").
#' @return A list containing two elements:
#'   - `all_args`: A named list of arguments for the new function signature,
#'     initialized with `rlang::zap()`.
#'   - `parsnip_names`: A character vector of all argument names for `parsnip`.
#' @noRd
collect_spec_args <- function(
  layer_blocks,
  global_args = c(
    "epochs",
    "batch_size",
    "learn_rate",
    "validation_split",
    "verbose",
    "compile_loss",
    "compile_optimizer",
    "compile_metrics"
  )
) {
  if (any(c("compile", "optimizer") %in% names(layer_blocks))) {
    stop(
      "`compile` and `optimizer` are protected names and cannot be used as layer block names.",
      call. = FALSE
    )
  }

  all_args <- list()
  parsnip_names <- character()

  # block repetition counts (e.g., num_dense)
  for (block in names(layer_blocks)) {
    num_name <- paste0("num_", block)
    all_args[[num_name]] <- rlang::zap()
    parsnip_names <- c(parsnip_names, num_name)
  }

  # These args are passed by the fit engine, not set by the user in the spec
  engine_args <- c("input_shape", "num_classes")
  # block-specific parameters (skip first 'model' formal)
  for (block in names(layer_blocks)) {
    fmls_to_process <- rlang::fn_fmls(layer_blocks[[block]])[-1]
    # Filter out arguments that are provided by the fitting engine
    for (arg in names(fmls_to_process[
      !names(fmls_to_process) %in% engine_args
    ])) {
      full <- paste0(block, "_", arg)
      all_args[[full]] <- rlang::zap()
      parsnip_names <- c(parsnip_names, full)
    }
  }

  # global training parameters
  for (g in global_args) {
    all_args[[g]] <- rlang::zap()
    parsnip_names <- c(parsnip_names, g)
  }

  list(all_args = all_args, parsnip_names = parsnip_names)
}
