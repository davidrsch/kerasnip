# kerasnip (development version)

# kerasnip 0.1.2

## Bug Fixes

- Fixed CRAN NOTE: "Re-building vignettes had CPU time 3.1 times elapsed time".

## New Features

- Added `tidy()` and `glance()` methods for fitted kerasnip models, providing layer summaries and final training metrics.
- Added `probably` to Suggests for conformal inference support.
- Added the `conformal_intervals` vignette demonstrating prediction intervals using conformal inference with kerasnip workflows.

## Bug Fixes

- Improved `step_collapse()` documentation and `tidy()` method output.

# kerasnip 0.1.1

## Bug Fixes

- Fixed `predict()` failing with "Model not registered" after saving and reloading a kerasnip workflow in a new R session (#38). `predict()` now automatically replays the parsnip registration from metadata stored on the spec — no manual step required after `bundle::unbundle()` or `readRDS()`.
- Fixed `get_keras_object()` returning bare class constructors instead of instances for loss and metric objects, which caused `save_model()` to fail when those objects were passed to `compile()` (#42).
- Fixed `predict()` and `keras_evaluate()` / `extract_keras_model()` silently failing when the Python external pointer became invalid after an RDS round-trip. Both functions now detect the invalid pointer via `reticulate::py_validate_xptr()` and transparently restore the model from the serialized bytes stored in the fit object (#42).
- Fixed CRAN NOTE: added missing `importFrom(stats, predict)` so `predict.kerasnip_model_fit` is correctly resolved from the `stats` namespace.
- Fixed `compile_keras_grid()` crashing with a `vctrs_error_subscript_oob` error when passed a zero-row tibble (e.g. `tibble::tibble()`). The function now stops early with an informative message. Use `tibble::tibble(.rows = 1L)` to build the model once with the spec's current arguments and no hyperparameter variation.

## New Features

- Every spec instance now carries the `kerasnip_spec` class and embedded metadata (`kerasnip_layer_blocks`, `kerasnip_functional`), enabling transparent auto-registration on predict (closes #39).
- `fit()` on a kerasnip spec now tags the result with `kerasnip_model_fit` class to enable the auto-registration dispatch.
- At fit time the Keras model is serialized to a raw byte vector (`.keras` format) stored in the `model_fit` object. This makes plain `saveRDS()` / `readRDS()` fully supported without any extra steps (#42).
- `bundle::bundle()` / `unbundle()` is now also supported as an alternative persistence strategy for MLOps and deployment workflows (#42).

## Documentation

- Added the `saving_and_reloading` vignette explaining both the `saveRDS` and `bundle` workflows, with a comparison table and a description of the auto-restore mechanism (#42).
- Corrected the "Save and Reload" sections in the Sequential Workflows and Functional Workflows vignettes, which previously stated that `saveRDS` does not work (#42).
- Added a `@section` to both spec function reference pages explaining the `bundle::bundle()` workflow (closes #40).

# kerasnip 0.1.0

## Breaking changes

- `inp_spec()` now interprets named vectors in an argument-first orientation (`c(input_a = "processed_1")`). Existing code that used the previous upstream-first style must swap the names and values.

## Documentation

- Updated README, vignettes, and reference docs to reflect the new `inp_spec()` mapping semantics and added guidance for migrating older code.

## Testing

- Added a regression test that fails fast when the legacy mapping orientation is supplied.

# kerasnip 0.0.3

## Improvements

- Added comprehensive end-to-end tests for tuning `fit_*` and `compile_*` parameters, and for `autoplot` uniqueness with multiple similar parameters.

## Documentation

- Added new vignettes:
  - "Transfer Learning with Keras Applications"
  - "Tuning Multiple Similar Parameters: Ensuring `autoplot` Uniqueness"
  - "Tuning Fit and Compile Arguments"

## Bug fixes

- Enhanced `register_model_args` to improve matching of Keras arguments to `dials` functions and correctly assign package sources for `dials` parameters.
- Refined `remove_keras_spec` to be more precise in removing model specifications, preventing unintended removal of other objects.

# kerasnip 0.0.2

## Improvements

- Test suite improvements for post-processing and fit helpers (#23).

## Bug Fixes

- Fixed a bug in the documentation where examples were not self-contained, causing issues with CRAN checks. This involved updating examples to be fully runnable and cleaning up created model specifications (#22).
- As part of this fix, a new helper function `model_exists()` was introduced and exported.

# kerasnip 0.0.1

## New features

- Added support for **functional API** (`create_keras_functional_spec()`) (#6).
- Introduced **custom steps**, including `step_collapse` for collapsing columns in list columns (#20).
- Added **evaluation helpers**: `keras_evaluate()`, extractors for summary and history (#12).
- Introduced modularized helpers for **build and compile** of keras models (#14).
- Added **sequential workflow** and **functional workflow** examples and vignettes (#20).
- Added new **tests** to improve coverage.
- Added **pkgdown site improvements** (favicon, documentation pages, guides) (#10, #16, #18 and #20).

## Improvements

- Refactored code for modularity and clarity.
- Updated documentation across multiple guides and functions.
- Improved consistency with **tidy naming conventions**.
- Improved robustness of tests and error handling.

## Bug fixes

- Fixed issues with **compile\_** and **fit\_** argument handling.
- Fixed issues with `predict()` and `evaluate()` to handle multiple outputs correctly (#18).
- Fixed documentation typos and pkgdown errors.
- Fixed utils issues and missing dependencies.
- Fixed warnings and CRAN check issues.

## Breaking changes

- Changed `fit` interface to use formula, supporting list columns (#18).

# kerasnip 0.0.0.9000

- Initial development version.
- Added `create_keras_spec()` to generate `parsnip` specifications dynamically.
