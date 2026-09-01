# kerasnip (development version)

## New Features

- Fit arguments can now be written without the `fit_` prefix (e.g. `epochs` for `fit_epochs`), matching other tidymodels neural-network engines. The `fit_` form takes precedence when both are supplied.
- Added `step_sequence()` and `step_lead()` recipe steps for building the windowed inputs and multi-step-ahead targets a recurrent forecasting model needs.
- Added support for multi-step (vector-valued) regression outputs: a single `output` block can now predict several future steps at once (`units = horizon`), with `predict()` returning point forecasts and `conf_int`/`pred_int` nested by forecast step.
- Added `predict(..., type = "pred_int", joint = TRUE)` for correlated multi-step prediction intervals, sampled from the jointly-estimated residual covariance across steps and returned as tidybayes-style `.draw` columns.
- Added the `multistep_forecasting` vignette demonstrating LSTM-based multi-step forecasting end to end, including uncertainty intervals.
- Added `tailor` to Suggests and verified `tailor`/`probably` post-processing (calibration, probability thresholding, conformal inference) works on kerasnip's standard single-output prediction tibbles.
- Added `kerasnip_output_view()`, presenting one output of a multi-output fit as an ordinary single-output fit (`predict()` returns plain `.pred`/`.pred_class`/`.pred_<level>` columns), so it can be calibrated with `tailor`/`probably` one output at a time. `probably::int_conformal_split()` and `probably::int_conformal_full()` work directly against a view; `int_conformal_full()`'s refit loop substitutes the other output(s)' own point-prediction as a placeholder for the (otherwise unobserved) value on each candidate row — a reasonable but unproven assumption, documented on the function.
- Added `kerasnip_step_view()` and `kerasnip_step_truth()` for the same purpose on multistep forecasting models: flattens one forecast step's nested `.pred` entry down to a standard `.pred` column, and recovers the true future value at that step by re-baking the fitted recipe (per-step outcome columns are recipe-engineered via `step_lead()`, not present in raw data). `probably::int_conformal_split()` and `probably::int_conformal_full()` both work directly against a step view. `int_conformal_full()`'s refit loop needs a candidate value written into the single *raw* column `step_lead()` derives every step's truth from, which also shifts every other step's target forecast from the same origin — those other steps' placeholders come from the current model's own forecast (the same idea as the multi-output case). Only supported when `step_lead()` and `step_sequence()` share a single source column, true of every multistep model this package's own examples build.
- Added `kerasnip_add_tailor()`, a `workflows::add_tailor()`-alike for multi-output and multistep models: attaches a `tailor` post-processor to a single output or forecast step and splices the adjusted values back into the full prediction, leaving every other output/step untouched. `workflows::add_tailor()` itself cannot be used directly on these models — `tailor::fit()` requires a single, flat, numeric outcome/estimate column pair, which neither shape provides.

## Bug Fixes

- Fixed multi-output classification (multiple factor targets, each its own head) failing when fit through a `workflow()`. parsnip's internal formula reconstruction rebuilds multi-column outcomes via `cbind()`, which silently discards factor levels; the fit interface is now `"data.frame"` instead of `"formula"` to avoid this.
- Fixed `predict(..., type = "class")` erroring for multi-output classification models.
- Fixed `conf_int`/`pred_int` for multi-output classification returning garbled column names; factor levels are now sliced per output instead of passing the whole named list to every output.
- Fixed multi-output regression predictions keeping `matrix`/`array` class on a single-row `.pred_<output>` column instead of a plain numeric (`tibble::as_tibble()` on a list of single-column matrices doesn't simplify a 1-row matrix element the way it does for more rows). This silently broke anything re-using a single-row prediction as a plain value, such as `kerasnip_output_view()`'s `probably::int_conformal_full()` support.

## Documentation

- Corrected model-spec examples and wording to use `fit_epochs` consistently with the generated arguments.
- Fixed a typo in the `keras_evaluate()` description and a typo in the pkgdown reference index.

## Testing

- Added an end-to-end integration test for `stacks` ensembles (#48) and added `stacks` to `Suggests`.

## Maintenance

- Aligned code style with the Air formatter and tidyverse conventions.

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
