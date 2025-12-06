# Compile Keras Models Over a Grid of Hyperparameters

Pre-compiles Keras models for each hyperparameter combination in a grid.

This function is a powerful debugging tool to use before running a full
[`tune::tune_grid()`](https://tune.tidymodels.org/reference/tune_grid.html).
It allows you to quickly validate multiple model architectures, ensuring
they can be successfully built and compiled without the time-consuming
process of actually fitting them. It helps catch common errors like
incompatible layer shapes or invalid argument values early.

## Usage

``` r
compile_keras_grid(spec, grid, x, y)
```

## Arguments

- spec:

  A `parsnip` model specification created by
  [`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_sequential_spec.md)
  or
  [`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_functional_spec.md).

- grid:

  A `tibble` or `data.frame` containing the grid of hyperparameters to
  evaluate. Each row represents a unique model architecture to be
  compiled.

- x:

  A data frame or matrix of predictors. This is used to infer the
  `input_shape` for the Keras model.

- y:

  A vector or factor of outcomes. This is used to infer the output shape
  and the default loss function for the Keras model.

## Value

A `tibble` with the following columns:

- Columns from the input `grid`.

- `compiled_model`: A list-column containing the compiled Keras model
  objects. If compilation failed, the element will be `NULL`.

- `error`: A list-column containing `NA` for successes or a character
  string with the error message for failures.

## Details

Compile and Validate Keras Model Architectures

The function iterates through each row of the provided `grid`. For each
hyperparameter combination, it attempts to build and compile the Keras
model defined by the `spec`. The process is wrapped in a `try-catch`
block to gracefully handle and report any errors that occur during model
instantiation or compilation.

The output is a tibble that mirrors the input `grid`, with additional
columns containing the compiled model object or the error message,
making it easy to inspect which architectures are valid.

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
  model_name = "my_mlp_grid",
  layer_blocks = list(
    input = input_block,
    hidden = hidden_block,
    output = output_block
  ),
  mode = "classification"
)

mlp_spec <- my_mlp_grid(
  hidden_units = tune(),
  compile_loss = "categorical_crossentropy",
  compile_optimizer = "adam"
)

# 3. Create a hyperparameter grid
# Include an invalid value (-10) to demonstrate error handling
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

print(compiled_grid)
remove_keras_spec("my_mlp_grid")

# 6. Inspect the results
# The row with `hidden_units = -10` will show an error.
}
#> 
#> Attaching package: ‘parsnip’
#> The following object is masked from ‘package:kerasnip’:
#> 
#>     get_model_env
#> Loading required package: scales
#> # A tibble: 3 × 3
#>   hidden_units compiled_model                           error                   
#>          <dbl> <list>                                   <chr>                   
#> 1           32 <keras.src.models.sequential.Sequential>  NA                     
#> 2           64 <keras.src.models.sequential.Sequential>  NA                     
#> 3          -10 <NULL>                                   "ValueError: Cannot con…
#> Removed from parsnip registry objects: my_mlp_grid, my_mlp_grid_args, my_mlp_grid_encoding, my_mlp_grid_fit, my_mlp_grid_modes, my_mlp_grid_pkgs, my_mlp_grid_predict
#> Removed 'my_mlp_grid' from parsnip:::get_model_env()$models
# }
```
