# Inform About Compilation Errors

This helper function inspects the results from
[`compile_keras_grid()`](https://davidrsch.github.io/kerasnip/reference/compile_keras_grid.md)
and prints a formatted, easy-to-read summary of any compilation errors
that occurred.

## Usage

``` r
inform_errors(compiled_grid, n = 10)
```

## Arguments

- compiled_grid:

  A tibble, the result of a call to
  [`compile_keras_grid()`](https://davidrsch.github.io/kerasnip/reference/compile_keras_grid.md).

- n:

  A single integer for the maximum number of distinct errors to display
  in detail.

## Value

Invisibly returns the input `compiled_grid`. Called for its side effect
of printing a summary to the console.

## Details

Display a Summary of Compilation Errors

This is most useful for interactive debugging of complex tuning grids
where some hyperparameter combinations may lead to invalid Keras models.

## Examples

``` r
# \donttest{
if (requireNamespace("keras3", quietly = TRUE)) {
  library(keras3)
  library(parsnip)
  library(dials)

  # 1. Define layer blocks
  input_block <- function(model, input_shape) {
    keras_model_sequential(input_shape = input_shape)
  }
  hidden_block <- function(model, units = 32) {
    model |> layer_dense(units = units, activation = "relu")
  }
  output_block <- function(model, num_classes) {
    model |> layer_dense(units = num_classes, activation = "softmax")
  }

  # 2. Define a kerasnip model specification
  create_keras_sequential_spec(
    model_name = "my_mlp_grid_3",
    layer_blocks = list(
      input = input_block,
      hidden = hidden_block,
      output = output_block
    ),
    mode = "classification"
  )

  mlp_spec <- my_mlp_grid_3(
    hidden_units = tune(),
    compile_loss = "categorical_crossentropy",
    compile_optimizer = "adam"
  )

  # 3. Create a hyperparameter grid
  param_grid <- tibble::tibble(
    hidden_units = c(32, 64, -10)
  )

  # 4. Prepare dummy data
  x_train <- matrix(rnorm(100 * 10), ncol = 10)
  y_train <- factor(sample(0:1, 100, replace = TRUE))

  # 5. Compile models over the grid
  compiled_grid <- compile_keras_grid(
    spec = mlp_spec,
    grid = param_grid,
    x = x_train,
    y = y_train
  )

  # 6. Inform about errors
  inform_errors(compiled_grid)
  remove_keras_spec("my_mlp_grid_3")
}
#> 
#> ── Compilation Errors Summary ──────────────────────────────────────────────────
#> ✖ 1 of 3 models failed to compile.
#> 
#> ── Error 1/1 ──
#> 
#> Hyperparameters:
#> hidden_units: -10
#> Error Message:
#> ValueError: Cannot convert '(10, -10)' to a shape. Negative dimensions are not allowed.
#> Run `reticulate::py_last_error()` for details.
#> Removed from parsnip registry objects: my_mlp_grid_3, my_mlp_grid_3_args, my_mlp_grid_3_encoding, my_mlp_grid_3_fit, my_mlp_grid_3_modes, my_mlp_grid_3_pkgs, my_mlp_grid_3_predict
#> Removed 'my_mlp_grid_3' from parsnip:::get_model_env()$models
# }
```
