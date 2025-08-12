#' Register Fit and Prediction Methods with Parsnip
#'
#' @description
#' This function registers the methods that `parsnip` will use to fit the model
#' and generate predictions.
#'
#' @details
#' This function makes calls to `parsnip::set_fit()` and `parsnip::set_pred()`:
#' - `set_fit()`: Links the model specification to the appropriate generic
#'   fitting engine (`generic_sequential_fit()` or `generic_functional_fit()`).
#'   It also passes the user's `layer_blocks` list as a default argument to
#'   the fitting function.
#' - `set_pred()`: Defines how to generate predictions for different types
#'   ("numeric", "class", "prob"). It specifies the underlying `predict()`
#'   method and the post-processing functions (`keras_postprocess_*`) needed
#'   to format the output into a standardized `tidymodels` tibble.
#'
#' @param model_name The name of the new model.
#' @param mode The model mode ("regression" or "classification").
#' @param layer_blocks The named list of layer block functions, which is passed
#'   as a default argument to the fit function.
#' @param functional A logical. If `TRUE`, registers `generic_functional_fit` as
#'   the fitting engine. Otherwise, registers `generic_sequential_fit`.
#' @return Invisibly returns `NULL`. Called for its side effects.
#' @noRd
register_fit_predict <- function(model_name, mode, layer_blocks, functional) {
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
          "generic_functional_fit"
        } else {
          "generic_sequential_fit"
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
          x = rlang::expr(process_x(new_data)$x_proc)
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
          x = rlang::expr(process_x(new_data)$x_proc)
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
          x = rlang::expr(process_x(new_data)$x_proc)
        )
      )
    )
  }
}

#' Post-process Keras Numeric Predictions
#'
#' @description
#' Formats raw numeric predictions from a Keras model into a tibble with the
#' standardized `.pred` column, as required by `tidymodels`.
#'
#' @details
#' This function simply takes the matrix output from `keras3::predict()` and
#' converts it to a single-column tibble.
#' @param results A matrix of numeric predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with a `.pred` column.
#' @noRd
keras_postprocess_numeric <- function(results, object) {
  tibble::tibble(.pred = as.vector(results))
}

#' Post-process Keras Probability Predictions
#'
#' @description
#' Formats raw probability predictions from a Keras model into a tibble with
#' class-specific column names (e.g., `.pred_class1`, `.pred_class2`).
#'
#' @details
#' This function retrieves the original factor levels from `object$fit$lvl`
#' (which was stored by the fitting engine) and uses them to name the columns.
#' @param results A matrix of probability predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with named columns for each class probability.
#' @noRd
keras_postprocess_probs <- function(results, object) {
  # The levels are now nested inside the fit object
  colnames(results) <- object$fit$lvl
  tibble::as_tibble(results)
}

#' Post-process Keras Class Predictions
#'
#' @description
#' Converts raw probability predictions from a Keras model into a single
#' `.pred_class` column of factor predictions.
#'
#' @details
#' For multiclass models, it finds the class with the highest probability
#' (`which.max`). For binary models, it applies a 0.5 threshold. It uses the
#' levels stored in `object$fit$lvl` to ensure the output factor is correct.
#' @param results A matrix of probability predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with a `.pred_class` column containing factor predictions.
#' @noRd
keras_postprocess_classes <- function(results, object) {
  # The levels are now nested inside the fit object
  lvls <- object$fit$lvl
  if (ncol(results) == 1) {
    # Binary classification
    pred_class <- ifelse(results[, 1] > 0.5, lvls[2], lvls[1])
    pred_class <- factor(pred_class, levels = lvls)
  } else {
    # Multiclass classification
    pred_class_int <- apply(results, 1, which.max)
    pred_class <- lvls[pred_class_int]
    pred_class <- factor(pred_class, levels = lvls)
  }
  tibble::tibble(.pred_class = pred_class)
}
