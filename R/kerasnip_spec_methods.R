#' set_args Method for kerasnip Spec Objects
#'
#' @description
#' S3 method for `set_args()` dispatched on `kerasnip_spec` objects.
#' `parsnip::set_args.model_spec()` calls `new_model_spec()`, which strips any
#' extra classes and attributes. This wrapper saves and re-attaches the
#' `kerasnip_layer_blocks` and `kerasnip_functional` metadata attributes (and
#' the `kerasnip_spec` class) after `NextMethod()` has done its work.
#'
#' @param object A `kerasnip_spec` model specification.
#' @param ... Named model arguments to update, passed to
#'   `parsnip::set_args.model_spec()`.
#' @return A `model_spec` object with the `kerasnip_spec` class and metadata
#'   attributes re-attached.
#' @keywords internal
#' @importFrom parsnip set_args
#' @exportS3Method parsnip::set_args
set_args.kerasnip_spec <- function(object, ...) {
  layer_blocks <- attr(object, "kerasnip_layer_blocks")
  functional <- attr(object, "kerasnip_functional")
  result <- NextMethod()
  class(result) <- c(class(result)[1L], "kerasnip_spec", class(result)[-1L])
  attr(result, "kerasnip_layer_blocks") <- layer_blocks
  attr(result, "kerasnip_functional") <- functional
  result
}

#' set_engine Method for kerasnip Spec Objects
#'
#' @description
#' S3 method for `set_engine()` dispatched on `kerasnip_spec` objects.
#' `parsnip::set_engine.model_spec()` internally calls `new_model_spec()`,
#' which re-creates the spec from scratch with only `c(cls, "model_spec")` as
#' the class vector — stripping `kerasnip_spec` and any custom attributes. This
#' wrapper preserves the `kerasnip_layer_blocks` and `kerasnip_functional`
#' metadata attributes and re-attaches them (along with the `kerasnip_spec`
#' class) after `NextMethod()` has done its work.
#'
#' @param object A `kerasnip_spec` model specification.
#' @param engine A character string naming the engine (e.g., `"keras"`).
#' @param ... Additional engine-specific arguments passed to
#'   `parsnip::set_engine.model_spec()`.
#' @return A `model_spec` object with the `kerasnip_spec` class and metadata
#'   attributes re-attached.
#' @keywords internal
#' @importFrom parsnip set_engine
#' @exportS3Method parsnip::set_engine
set_engine.kerasnip_spec <- function(object, engine, ...) {
  layer_blocks <- attr(object, "kerasnip_layer_blocks")
  functional <- attr(object, "kerasnip_functional")
  result <- NextMethod()
  class(result) <- c(class(result)[1L], "kerasnip_spec", class(result)[-1L])
  attr(result, "kerasnip_layer_blocks") <- layer_blocks
  attr(result, "kerasnip_functional") <- functional
  result
}

#' Fit Method for kerasnip Spec Objects
#'
#' @description
#' S3 method for `fit()` dispatched on `kerasnip_spec` objects. Delegates to
#' the standard parsnip `fit.model_spec()` and then tags the result with the
#' `kerasnip_model_fit` class so that `predict.kerasnip_model_fit()` is
#' dispatched on subsequent calls.
#'
#' `kerasnip_spec` is stripped from the class before `NextMethod()` to prevent
#' parsnip's internal `specific_model()` helper from returning more than one
#' model-class entry, which would break registry lookups. The custom metadata
#' attributes remain on the object and are thus stored inside the resulting
#' `model_fit$spec`.
#'
#' @param object A `kerasnip_spec` model specification.
#' @param ... Passed to `parsnip::fit.model_spec()`.
#' @return A `model_fit` object with the additional `kerasnip_model_fit` class
#'   prepended to its class vector.
#' @keywords internal
#' @importFrom generics fit
#' @exportS3Method generics::fit
fit.kerasnip_spec <- function(object, ...) {
  class(object) <- class(object)[class(object) != "kerasnip_spec"]
  result <- NextMethod()
  class(result) <- c("kerasnip_model_fit", class(result))
  result
}

#' fit_xy Method for kerasnip Spec Objects
#'
#' @description
#' S3 method for `fit_xy()` dispatched on `kerasnip_spec` objects. Workflows
#' route through `fit_xy` rather than `fit`, so this method ensures the
#' `kerasnip_model_fit` class is attached in the workflow fitting path as well.
#'
#' `kerasnip_spec` is stripped from the class before `NextMethod()` to prevent
#' parsnip's internal `specific_model()` helper from returning more than one
#' model-class entry, which would break registry lookups. The custom metadata
#' attributes remain on the object and are thus stored inside the resulting
#' `model_fit$spec`.
#'
#' @param object A `kerasnip_spec` model specification.
#' @param ... Passed to `parsnip::fit_xy.model_spec()`.
#' @return A `model_fit` object with the additional `kerasnip_model_fit` class
#'   prepended to its class vector.
#' @keywords internal
#' @importFrom generics fit_xy
#' @exportS3Method generics::fit_xy
fit_xy.kerasnip_spec <- function(object, ...) {
  class(object) <- class(object)[class(object) != "kerasnip_spec"]
  result <- NextMethod()
  class(result) <- c("kerasnip_model_fit", class(result))
  result
}

#' Predict Method for kerasnip Model Fits
#'
#' @description
#' S3 method for `predict()` dispatched on `kerasnip_model_fit` objects.
#' Before delegating to the standard parsnip predict machinery, it checks
#' whether the underlying model type is registered in the current parsnip
#' session. If not (e.g. after loading a saved workflow in a new R session),
#' it transparently replays the full parsnip registration using metadata stored
#' on the spec object — requiring no manual step from the user.
#'
#' @details
#' The metadata needed for re-registration (`kerasnip_layer_blocks`,
#' `kerasnip_functional`) is embedded on the spec object by the spec
#' constructor function at call time. This means it is preserved across
#' `saveRDS()`/`readRDS()` and `bundle()`/`unbundle()` round-trips.
#'
#' For full model weight portability (i.e. to be able to `predict()` on new
#' data in a new R session), use `bundle::bundle()` before saving. Plain
#' `saveRDS()` preserves the spec structure and will auto-register, but the
#' underlying Keras model weights are not portable without bundling.
#'
#' @param object A `kerasnip_model_fit` object.
#' @param new_data A data frame of predictors.
#' @param ... Passed to the parsnip predict method.
#' @return A tibble of predictions.
#' @keywords internal
#' @exportS3Method stats::predict
predict.kerasnip_model_fit <- function(object, new_data, ...) {
  model_name <- class(object$spec)[1L]

  if (!model_exists(model_name)) {
    spec <- object$spec
    layer_blocks <- attr(spec, "kerasnip_layer_blocks")
    functional <- attr(spec, "kerasnip_functional") %||% FALSE
    mode <- spec$mode
    args_info <- collect_spec_args(layer_blocks, functional)

    register_core_model(model_name, mode)
    register_model_args(model_name, args_info$parsnip_names)
    register_fit_predict(model_name, mode, layer_blocks, functional)
    register_update_method(
      model_name,
      args_info$parsnip_names,
      env = globalenv()
    )
  }

  NextMethod()
}
