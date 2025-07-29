#' Register Model Arguments with Parsnip
#'
#' Registers each model argument with `parsnip` and maps it to a corresponding
#' `dials` parameter function for tuning. This allows `tidymodels` to know
#' about the tunable parameters of the custom model.
#'
#' @param model_name The name of the new model.
#' @param parsnip_names A character vector of all argument names.
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
    "epochs",
    "epochs",
    "batch_size",
    "batch_size",
    "compile_loss", # parsnip arg
    "loss_function_keras", # dials function from kerasnip
    "compile_optimizer", # parsnip arg
    "optimizer_function" # dials function from kerasnip
  )

  # We now allow optimizer to be tuned. Metrics are for tracking, not training.
  non_tunable <- c("verbose")

  for (arg in parsnip_names) {
    if (arg %in% non_tunable) {
      next
    }

    if (startsWith(arg, "num_")) {
      dials_fun <- "num_terms"
    } else {
      base_arg <- sub(".*_", "", arg)
      idx <- match(base_arg, keras_dials_map$keras_arg)
      dials_fun <- if (!is.na(idx)) keras_dials_map$dials_fun[idx] else arg
    }

    parsnip::set_model_arg(
      model = model_name,
      eng = "keras",
      parsnip = arg,
      original = arg,
      func = list(pkg = "dials", fun = dials_fun),
      has_submodel = FALSE
    )
  }
}