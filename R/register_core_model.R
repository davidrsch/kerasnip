#' Register Core Model Information with Parsnip
#'
#' Sets up the basic model definition with `parsnip`, including its mode,
#' engine, dependencies, and data encoding requirements.
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