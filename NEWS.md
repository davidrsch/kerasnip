# kerasnip (development version)

# kerasnip 0.0.2

## Improvements
* Test suite improvements for post-processing and fit helpers (#23).

## Bug Fixes
* Fixed a bug in the documentation where examples were not self-contained, causing issues with CRAN checks. This involved updating examples to be fully runnable and cleaning up created model specifications (#22).
* As part of this fix, a new helper function `model_exists()` was introduced and exported.



# kerasnip 0.0.1

## New features
* Added support for **functional API** (`create_keras_functional_spec()`) (#6).
* Introduced **custom steps**, including `step_collapse` for collapsing columns in list columns (#20).
* Added **evaluation helpers**: `keras_evaluate()`, extractors for summary and history (#12).
* Introduced modularized helpers for **build and compile** of keras models (#14).
* Added **sequential workflow** and **functional workflow** examples and vignettes (#20).
* Added new **tests** to improve coverage.
* Added **pkgdown site improvements** (favicon, documentation pages, guides) (#10, #16, #18 and #20).

## Improvements
* Refactored code for modularity and clarity.
* Updated documentation across multiple guides and functions.
* Improved consistency with **tidy naming conventions**.
* Improved robustness of tests and error handling.

## Bug fixes
* Fixed issues with **compile_** and **fit_** argument handling.
* Fixed issues with `predict()` and `evaluate()` to handle multiple outputs correctly (#18).
* Fixed documentation typos and pkgdown errors.
* Fixed utils issues and missing dependencies.
* Fixed warnings and CRAN check issues.

## Breaking changes
* Changed `fit` interface to use formula, supporting list columns (#18).

# kerasnip 0.0.0.9000

* Initial development version.
* Added `create_keras_spec()` to generate `parsnip` specifications dynamically.
