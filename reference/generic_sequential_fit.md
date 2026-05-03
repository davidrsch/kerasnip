# Internal Fitting Engine for Sequential API Models

This function serves as the internal engine for fitting `kerasnip`
models that are based on the Keras sequential API. It is not intended to
be called directly by the user. The function is invoked by
[`parsnip::fit()`](https://generics.r-lib.org/reference/fit.html) when a
`kerasnip` sequential model specification is used.

## Usage

``` r
generic_sequential_fit(formula, data, layer_blocks, ...)
```

## Arguments

- formula:

  A formula specifying the predictor and outcome variables, passed down
  from the
  [`parsnip::fit()`](https://generics.r-lib.org/reference/fit.html)
  call.

- data:

  A data frame containing the training data, passed down from the
  [`parsnip::fit()`](https://generics.r-lib.org/reference/fit.html)
  call.

- layer_blocks:

  A named list of layer block functions. This is passed internally from
  the `parsnip` model specification.

- ...:

  Additional arguments passed down from the model specification. These
  can include:

  - **Layer Parameters:** Arguments for the layer blocks, prefixed with
    the block name (e.g., `dense_units = 64`).

  - **Architecture Parameters:** Arguments to control the number of
    times a block is repeated, in the format `num_{block_name}` (e.g.,
    `num_dense = 2`).

  - **Compile Parameters:** Arguments to customize model compilation,
    prefixed with `compile_` (e.g., `compile_loss = "mae"`,
    `compile_optimizer = "sgd"`).

  - **Fit Parameters:** Arguments to customize model fitting, prefixed
    with `fit_` (e.g., `fit_callbacks = list(...)`,
    `fit_class_weight = list(...)`).

## Value

A list containing the fitted model and other metadata. This list is
stored in the `fit` slot of the `parsnip` model fit object. The list
contains the following elements:

- `fit`: The raw, fitted Keras model object.

- `history`: The Keras training history object.

- `lvl`: A character vector of the outcome factor levels (for
  classification) or `NULL` (for regression).

## Details

Generic Fitting Function for Sequential Keras Models

The function orchestrates the three main steps of the model fitting
process:

1.  **Build and Compile:** It calls
    `build_and_compile_sequential_model()` to construct the Keras model
    architecture based on the provided `layer_blocks` and
    hyperparameters.

2.  **Process Data:** It preprocesses the input (`x`) and output (`y`)
    data into the format expected by Keras.

3.  **Fit Model:** It calls
    [`keras3::fit()`](https://generics.r-lib.org/reference/fit.html)
    with the compiled model and processed data, passing along any
    fitting-specific arguments (e.g., `epochs`, `batch_size`,
    `callbacks`).

## Examples

``` r
# This function is not called directly by users.
# It is called internally by `parsnip::fit()`.
# For example:
# \donttest{
library(parsnip)
# create_keras_sequential_spec(...) defines my_sequential_model

# spec <- my_sequential_model(hidden_1_units = 128, fit_epochs = 10) |>
#   set_engine("keras")

# # This call to fit() would invoke generic_sequential_fit() internally
# fitted_model <- fit(spec, y ~ x, data = training_data)
# }
```
