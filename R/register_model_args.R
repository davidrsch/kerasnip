#' Register Model Arguments with Parsnip and Dials
#'
#' @description
#' Registers each model argument with `parsnip` and maps it to a corresponding
#' `dials` parameter function. This is a crucial step that makes the model's
#' parameters visible to the `tidymodels` ecosystem for tuning.
#'
#' @details
#' This function iterates through each argument name discovered by
#' `collect_spec_args()` and calls `parsnip::set_model_arg()`.
#'
#' The mapping from a `kerasnip` argument to a `dials` function is determined
#' by the following logic:
#' \itemize{
#'   \item Arguments starting with `num_` (e.g., `num_dense`) are mapped to
#'     `dials::num_terms()`.
#'   \item Other arguments are mapped based on their suffix (e.g., `dense_units`
#'     is mapped based on `units`). The internal `keras_dials_map` object
#'     contains common mappings like `units` -> `dials::hidden_units()`.
#'   \item Arguments for `compile_loss` and `compile_optimizer` are mapped to custom
#'     `dials` parameter functions (`loss_function_keras()` and `optimizer_function()`)
#'     that are part of the `kerasnip` package itself. The function correctly
#'     sets the `pkg` for these to `kerasnip`.
#' }
#'
#' @param model_name The name of the new model specification.
#' @param parsnip_names A character vector of all argument names to be registered.
#' @return Invisibly returns `NULL`. Called for its side effects.
#' @noRd
register_model_args <- function(model_name, parsnip_names) {
  keras_dials_map <- tibble::tribble(
    ~keras_arg,
    ~dials_fun,
    "units",
    "hidden_units",
    "filters",
    "hidden_units",
    "kernel_size",
    "kernel_size",
    "pool_size",
    "pool_size",
    "dropout",
    "dropout",
    "rate",
    "dropout",
    "learn_rate",
    "learn_rate",
    "fit_epochs",
    "epochs",
    "fit_batch_size",
    "batch_size",
    "compile_loss", # parsnip arg
    "loss_function_keras", # dials function from kerasnip
    "compile_optimizer", # parsnip arg
    "optimizer_function" # dials function from kerasnip
  )

  # We now allow optimizer to be tuned. Metrics are for tracking, not training.
  non_tunable <- c("fit_verbose")

  for (arg in parsnip_names) {
    if (arg %in% non_tunable) {
      next
    }

    if (startsWith(arg, "num_")) {
      dials_fun <- "num_terms"
    } else {
      # First, try to match the full argument name
      idx <- match(arg, keras_dials_map$keras_arg)
      if (!is.na(idx)) {
        dials_fun <- keras_dials_map$dials_fun[idx]
      } else {
        # If no full match, try to match the base name (e.g., "units" from "dense_units")
        base_arg <- sub(".*_", "", arg)
        idx <- match(base_arg, keras_dials_map$keras_arg)
        dials_fun <- if (!is.na(idx)) keras_dials_map$dials_fun[idx] else arg
      }
    }

    pkg <- if (dials_fun %in% c("loss_function_keras", "optimizer_function")) {
      "kerasnip"
    } else {
      "dials"
    }

    parsnip::set_model_arg(
      model = model_name,
      eng = "keras",
      parsnip = arg,
      original = arg,
      func = list(pkg = pkg, fun = dials_fun),
      has_submodel = FALSE
    )
  }
}
