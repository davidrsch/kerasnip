#' Re-register a Kerasnip Model Specification with Parsnip
#'
#' @description
#' Restores the parsnip model registration for a kerasnip model specification
#' after it has been loaded into a new R session.
#'
#' @details
#' `create_keras_sequential_spec()` and `create_keras_functional_spec()` work by
#' registering a new model type with parsnip via side-effect calls to
#' `parsnip::set_new_model()`, `parsnip::set_fit()`, etc. These registrations are
#' stored in parsnip's internal session environment and are **not** serialized
#' when a model or workflow is saved with [saveRDS()] or `bundle::bundle()`.
#'
#' When a kerasnip workflow is loaded in a new R session, calling `predict()`
#' will fail with:
#' ```
#' Error in `get_encoding()`: ! Model "your_model" has not been registered.
#' ```
#'
#' `reregister_keras_spec()` solves this by reading registration metadata that
#' is embedded in every kerasnip spec object (since v0.1.1) and replaying the
#' parsnip registration calls. It also re-creates the named constructor function
#' (e.g. `basic_mlp()`) in the calling environment.
#'
#' ## Recommended save/load workflow
#'
#' ```r
#' # Session 1 — save
#' library(kerasnip)
#' library(bundle)
#'
#' create_keras_sequential_spec("basic_mlp", layer_blocks = ..., mode = "regression")
#' wf      <- workflow() |> add_model(basic_mlp()) |> add_formula(y ~ .)
#' fit_wf  <- fit(wf, data = train_data)
#' bundled <- bundle(fit_wf)
#' saveRDS(bundled, "model.rds")
#'
#' # Session 2 — load
#' library(kerasnip)
#' library(bundle)
#'
#' bundled  <- readRDS("model.rds")
#' fit_wf   <- unbundle(bundled)            # bundle restores the Keras model
#' reregister_keras_spec(fit_wf)            # kerasnip restores parsnip registration
#' predict(fit_wf, new_data = test_data)    # works
#' ```
#'
#' If you do not need to save the fitted Keras model weights (e.g. for testing
#' or sharing a spec only), you can skip `bundle` and use plain [saveRDS()]
#' combined with `reregister_keras_spec()` to restore the parsnip-only side.
#'
#' @param x A kerasnip model specification (an object whose class inherits from
#'   a registered kerasnip model), or a fitted `workflow` containing one.
#'   The object must carry `kerasnip_layer_blocks` and `kerasnip_functional`
#'   attributes — these are embedded automatically by
#'   `create_keras_sequential_spec()` / `create_keras_functional_spec()` in
#'   kerasnip >= 0.1.1.
#' @param env The environment in which to (re-)create the named constructor
#'   function (e.g. `basic_mlp`). Defaults to the calling environment.
#'
#' @return Invisibly returns the model name (a character string). Called
#'   primarily for its side effects.
#'
#' @seealso [create_keras_sequential_spec()], [create_keras_functional_spec()],
#'   [remove_keras_spec()]
#'
#' @export
reregister_keras_spec <- function(x, env = parent.frame()) {
  # Accept either a spec or a workflow containing one
  if (inherits(x, "workflow")) {
    if (!requireNamespace("workflows", quietly = TRUE)) {
      cli::cli_abort(
        "The {.pkg workflows} package is required to extract a spec from a workflow."
      )
    }
    spec <- workflows::extract_spec_parsnip(x)
  } else {
    spec <- x
  }

  layer_blocks <- attr(spec, "kerasnip_layer_blocks")
  functional <- attr(spec, "kerasnip_functional")

  if (is.null(layer_blocks) || is.null(functional)) {
    cli::cli_abort(c(
      "The spec object does not carry kerasnip re-registration metadata.",
      "i" = "This metadata is embedded by {.fn create_keras_sequential_spec} and
             {.fn create_keras_functional_spec} in kerasnip >= 0.1.1.",
      "i" = "If this spec was created with an older version of kerasnip, call
             {.fn create_keras_sequential_spec} or {.fn create_keras_functional_spec}
             again to re-register the model."
    ))
  }

  model_name <- class(spec)[1]
  mode <- spec$mode

  args_info <- collect_spec_args(layer_blocks, functional = functional)

  register_core_model(model_name, mode)
  register_model_args(model_name, args_info$parsnip_names)
  register_fit_predict(model_name, mode, layer_blocks, functional = functional)
  register_update_method(model_name, args_info$parsnip_names, env = env)

  spec_fun <- build_spec_function(
    model_name,
    mode,
    args_info$all_args,
    args_info$parsnip_names,
    layer_blocks,
    functional = functional
  )
  rlang::env_poke(env, model_name, spec_fun)

  invisible(model_name)
}
