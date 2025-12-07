# kerasnip

The goal of `kerasnip` is to provide a seamless bridge between the
`keras` and `tidymodels` frameworks. It allows for the dynamic creation
of `parsnip` model specifications for Keras models, making them fully
compatible with `tidymodels` workflows.

## Installation

You can install the development version of `kerasnip` from GitHub with:

``` r
# install.packages("pak")
pak::pak("davidrsch/kerasnip")
```

## Example

### Example 1: Building a Sequential MLP

This example shows the core workflow for building a simple, linear stack
of layers using
[`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/create_keras_sequential_spec.md).

``` r
library(kerasnip)
library(tidymodels)
library(keras3)

# 1. Define Keras layer blocks
# The first block initializes the model.
input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}
# Subsequent blocks add layers.
dense_block <- function(model, units = 32) {
  model |> layer_dense(units = units, activation = "relu")
}
# The final block creates the output layer.
output_block <- function(model) {
  model |>
    layer_dense(units = 1)
}

# 2. Create a spec from the layer blocks
# This creates a new model function, `basic_mlp()`, in your environment.
create_keras_sequential_spec(
  model_name = "basic_mlp",
  layer_blocks = list(
    input = input_block,
    dense = dense_block,
    output = output_block
  ),
  mode = "regression"
)

# 3. Use the generated spec to define a model.
# We can set the number of dense layers (`num_dense`) and their parameters (`dense_units`).
spec <- basic_mlp(
  num_dense = 2,
  dense_units = 64,
  fit_epochs = 10,
  learn_rate = 0.01
) |>
  set_engine("keras")

# 4. Fit the model within a tidymodels workflow
rec <- recipe(mpg ~ ., data = mtcars) |>
  step_normalize(all_numeric_predictors())

wf <- workflow(rec, spec)

set.seed(123)
fit_obj <- fit(wf, data = mtcars)

# 5. Make predictions
predict(fit_obj, new_data = mtcars[1:5, ])
#> # A tibble: 5 × 1
#>   .pred
#>   <dbl>
#> 1  21.3
#> 2  21.3
#> 3  22.8
#> 4  21.4
#> 5  18.7
```

### Example 2: Building a Functional “Fork-Join” Model

For complex, non-linear architectures, use
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/create_keras_functional_spec.md).
This example builds a model where the input is forked into two paths,
which are then concatenated.

``` r
library(kerasnip)
library(tidymodels)
library(keras3)

# 1. Define blocks. For the functional API, blocks are nodes in a graph.
input_block <- function(input_shape) layer_input(shape = input_shape)
path_block <- function(tensor, units = 16) tensor |> layer_dense(units = units)
concat_block <- function(input_a, input_b) layer_concatenate(list(input_a, input_b))
output_block <- function(tensor) layer_dense(tensor, units = 1)

# 2. Create the spec. The graph is defined by block names and their arguments.
create_keras_functional_spec(
  model_name = "forked_mlp",
  layer_blocks = list(
    main_input = input_block,
    path_a = inp_spec(path_block, "main_input"),
    path_b = inp_spec(path_block, "main_input"),
    concatenated = inp_spec(concat_block, c(input_a = "path_a", input_b = "path_b")),
    output = inp_spec(output_block, "concatenated")
  ),
  mode = "regression"
)

# 3. Use the new spec. Arguments are prefixed with their block name.
spec <- forked_mlp(path_a_units = 16, path_b_units = 8, fit_epochs = 10) |>
  set_engine("keras")

# Fit and predict as usual
set.seed(123)
fit(spec, mpg ~ ., data = mtcars) |>
  predict(new_data = mtcars[1:5, ])
#> # A tibble: 5 × 1
#>   .pred
#>   <dbl>
#> 1  19.4
#> 2  19.5
#> 3  21.9
#> 4  18.6
#> 5  17.9
```

### Example 3: Tuning a Sequential MLP Architecture

This example demonstrates how to tune the number of dense layers and the
rate of a final dropout layer, showcasing how to tune both architecture
and block hyperparameters simultaneously.

``` r
library(kerasnip)
library(tidymodels)
library(keras3)

# 1. Define Keras layer blocks for a tunable MLP
input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}
dense_block <- function(model, units = 32) {
  model |> layer_dense(units = units, activation = "relu")
}
dropout_block <- function(model, rate = 0.2) {
  model |> layer_dropout(rate = rate)
}
output_block <- function(model) {
  model |> layer_dense(units = 1)
}

# 2. Create a spec from the layer blocks
create_keras_sequential_spec(
  model_name = "tunable_mlp",
  layer_blocks = list(
    input = input_block,
    dense = dense_block,
    dropout = dropout_block,
    output = output_block
  ),
  mode = "regression"
)

# 3. Define a tunable model specification
tune_spec <- tunable_mlp(
  num_dense = tune(),
  dense_units = tune(),
  num_dropout = 1,
  dropout_rate = tune(),
  fit_epochs = 10
) |>
  set_engine("keras")

# 4. Set up and run a tuning workflow
rec <- recipe(mpg ~ ., data = mtcars) |>
  step_normalize(all_numeric_predictors())

wf_tune <- workflow(rec, tune_spec)

# Define the tuning grid.
params <- extract_parameter_set_dials(wf_tune) |>
  update(
    num_dense = dials::num_terms(c(1, 3)),
    dense_units = dials::hidden_units(c(8, 64)),
    dropout_rate = dials::dropout(c(0.1, 0.5))
  )
grid <- grid_regular(params, levels = 2)

# 5. Run the tuning
set.seed(456)
folds <- vfold_cv(mtcars, v = 3)

tune_res <- tune_grid(
  wf_tune,
  resamples = folds,
  grid = grid,
  control = control_grid(verbose = FALSE)
)

# 6. Show the best architecture
show_best(tune_res, metric = "rmse")
#> # A tibble: 5 × 7
#>   num_dense dense_units dropout_rate .metric .estimator .mean .config
#>       <int>       <int>        <dbl> <chr>   <chr>      <dbl> <chr>
#> 1         1          64          0.1 rmse    standard    2.92 Preprocessor1_Model02
#> 2         1          64          0.5 rmse    standard    3.02 Preprocessor1_Model08
#> 3         3          64          0.1 rmse    standard    3.15 Preprocessor1_Model04
#> 4         1           8          0.1 rmse    standard    3.20 Preprocessor1_Model01
#> 5         3           8          0.1 rmse    standard    3.22 Preprocessor1_Model03
```
