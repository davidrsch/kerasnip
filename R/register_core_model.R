#' Register Core Model Information with Parsnip
#'
#' @description
#' Sets up the basic model definition with `parsnip`. This function is called
#' once when a new specification is created.
#'
#' @details
#' This function makes a series of calls to `parsnip`'s registration API:
#' - `parsnip::set_new_model()`: Declares the new model.
#' - `parsnip::set_model_mode()`: Sets the mode (e.g., "regression").
#' - `parsnip::set_model_engine()`: Sets the engine to "keras".
#' - `parsnip::set_dependency()`: Declares the dependency on the `keras3` package.
#' - `parsnip::set_encoding()`: Specifies data preprocessing requirements.
#'
#' @param model_name The name of the new model.
#' @param mode The model mode ("regression" or "classification").
#' @return Invisibly returns `NULL`. Called for its side effects.
#' @noRd
register_core_model <- function(model_name, mode) {
  parsnip::set_new_model(model_name)
  parsnip::set_model_mode(model_name, mode)
  parsnip::set_model_engine(model_name, mode, "keras")
  parsnip::set_dependency(model_name, "keras", "keras3")

  parsnip::set_encoding(
    model = model_name,
    eng = "keras",
    mode = mode,
    options = list(
      predictor_indicators = "traditional",
      compute_intercept = TRUE,
      remove_intercept = TRUE,
      allow_sparse_x = FALSE
    )
  )
}
