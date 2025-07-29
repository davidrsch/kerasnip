#' Register Fit and Prediction Methods with Parsnip
#'
#' Defines how to fit the custom Keras model and how to generate predictions
#' for both regression and classification modes. It links the model to the
#' generic fitting implementation (`generic_keras_fit_impl`) and sets up
#' the appropriate prediction post-processing.
#'
#' @param model_name The name of the new model.
#' @param mode The model mode ("regression" or "classification").
#' @param layer_blocks The named list of layer block functions, which is passed
#'   as a default argument to the fit function.
#' @return Invisibly returns `NULL`. Called for its side effects.
#' @param functional A logical, if TRUE uses `generic_keras_functional_fit_impl` to fit, otherwise `generic_keras_fit_impl`. Defaults to FALSE.
#' @noRd
register_fit_predict <- function(model_name, mode, layer_blocks) {
  # Fit method
  parsnip::set_fit(
    model = model_name,
    eng = "keras",
    mode = mode,
    value = list(
      interface = "data.frame",
      protect = c("x", "y"),
      func = c(
        pkg = "kerasnip",
        fun = if (functional) {
          "generic_keras_functional_fit_impl"
        } else {
          "generic_keras_fit_impl"
        }
      ),
      defaults = list(layer_blocks = layer_blocks)
    )
  )

  # Regression prediction
  if (mode == "regression") {
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "regression",
      type = "numeric",
      value = list(
        pre = NULL,
        post = keras_postprocess_numeric,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = rlang::expr(as.matrix(new_data))
        )
      )
    )
  } else {
    # Classification predictions
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "classification",
      type = "class",
      value = list(
        pre = NULL,
        post = keras_postprocess_classes,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = rlang::expr(as.matrix(new_data))
        )
      )
    )
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "classification",
      type = "prob",
      value = list(
        pre = NULL,
        post = keras_postprocess_probs,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = rlang::expr(as.matrix(new_data))
        )
      )
    )
  }
}