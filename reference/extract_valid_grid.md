# Extract Valid Grid from Compilation Results

This helper function filters the results from
[`compile_keras_grid()`](https://davidrsch.github.io/kerasnip/reference/compile_keras_grid.md)
to return a new hyperparameter grid containing only the combinations
that compiled successfully.

## Usage

``` r
extract_valid_grid(compiled_grid)
```

## Arguments

- compiled_grid:

  A tibble, the result of a call to
  [`compile_keras_grid()`](https://davidrsch.github.io/kerasnip/reference/compile_keras_grid.md).

## Value

A tibble containing the subset of the original grid that resulted in a
successful model compilation. The `compiled_model` and `error` columns
are removed, leaving a clean grid ready for tuning.

## Details

Filter a Grid to Only Valid Hyperparameter Sets

After running
[`compile_keras_grid()`](https://davidrsch.github.io/kerasnip/reference/compile_keras_grid.md),
you can use this function to remove problematic hyperparameter
combinations before proceeding to the full
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html).

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
    model_name = "my_mlp_grid_2",
    layer_blocks = list(
      input = input_block,
      hidden = hidden_block,
      output = output_block
    ),
    mode = "classification"
  )

  mlp_spec <- my_mlp_grid_2(
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

  # 6. Extract the valid grid
  valid_grid <- extract_valid_grid(compiled_grid)
  print(valid_grid)
  remove_keras_spec("my_mlp_grid_2")
}
#> # A tibble: 2 × 1
#>   hidden_units
#>          <dbl>
#> 1           32
#> 2           64
#> Removed from parsnip registry objects: my_mlp_grid_2, my_mlp_grid_2_args, my_mlp_grid_2_encoding, my_mlp_grid_2_fit, my_mlp_grid_2_modes, my_mlp_grid_2_pkgs, my_mlp_grid_2_predict
#> Removed 'my_mlp_grid_2' from parsnip:::get_model_env()$models
# }
```
