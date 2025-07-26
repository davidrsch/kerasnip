
# kerasnip

<!-- badges: start -->
[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![R-CMD-check](https://github.com/davidrsch/kerasnip/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/davidrsch/kerasnip/actions/workflows/R-CMD-check.yaml)
<!-- badges: end -->

The goal of `kerasnip` is to provide a seamless bridge between the `keras` and `tidymodels` ecosystems. It allows for the dynamic creation of `parsnip` model specifications for Keras models, making them fully compatible with `tidymodels` workflows.

## Installation

You can install the development version of `kerasnip` from GitHub with:

```r
# install.packages("pak")
pak::pak("davidrsch/kerasnip")
```

## Example

### Example: Building a Sequential MLP from Layer Blocks

This example shows the core `kerasnip` workflow for building a model from modular "layer blocks". We will:
1. Define reusable blocks of Keras layers.
2. Create a model specification from these blocks.
3. Fit the model with a fixed architecture.

```r
library(kerasnip)
library(tidymodels)
library(keras3)

# 1. Define Keras layer blocks
# Each block is a function that takes a Keras model object and adds layers.
# The first block in the sequence is responsible for initializing the model.
mlp_input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}

mlp_dense_block <- function(model, units = 32) {
  model |>
    layer_dense(units = units, activation = "relu")
}

mlp_output_block <- function(model) {
  model |>
    layer_dense(units = 1)
}

# 2. Create a spec from the layer blocks
# This creates a new model function, `basic_mlp()`, in your environment.
create_keras_spec(
  model_name = "basic_mlp",
  layer_blocks = list(
    input = mlp_input_block,
    dense = mlp_dense_block,
    output = mlp_output_block
  ),
  mode = "regression"
)

# 3. Use the generated spec to define and fit a model
# We can set the number of dense layers (`num_dense`) and their parameters
# (`dense_units`).
spec <- basic_mlp(
  num_dense = 2,
  dense_units = 64,
  epochs = 50,
  learn_rate = 0.01
) |>
  set_engine("keras")

# 4. Fit the model within a tidymodels workflow
rec <- recipe(mpg ~ ., data = mtcars) |>
  step_normalize(all_numeric_predictors())

wf <- workflow() |>
  add_recipe(rec) |>
  add_model(spec)

set.seed(123)
fit_obj <- fit(wf, data = mtcars)

# 5. Make predictions
predictions <- predict(fit_obj, new_data = mtcars[1:5, ])
print(predictions)
#> # A tibble: 5 × 1
#>   .pred
#>   <dbl>
#> 1  22.6
#> 2  20.9
#> 3  26.1
#> 4  19.7
#> 5  17.8
```

### Example: Tuning a Sequential MLP Architecture

This example demonstrates how to tune the number of dense layers and the rate of a final dropout layer, showcasing how to tune both architecture and block hyperparameters simultaneously.

```r
library(kerasnip)
library(tidymodels)
library(keras3)

# 1. Define Keras layer blocks for a tunable MLP
mlp_input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}

tunable_dense_block <- function(model, units = 32) {
  model |> layer_dense(units = units, activation = "relu")
}

tunable_dropout_block <- function(model, rate = 0.2) {
  model |> layer_dropout(rate = rate)
}

mlp_output_block <- function(model) {
  model |> layer_dense(units = 1)
}

# 2. Create a spec from the layer blocks
create_keras_spec(
  model_name = "tunable_mlp",
  layer_blocks = list(
    input = mlp_input_block,
    dense = tunable_dense_block,
    dropout = tunable_dropout_block,
    output = mlp_output_block
  ),
  mode = "regression"
)

# 3. Define a tunable model specification
tune_spec <- tunable_mlp(
  num_dense = tune(),
  dense_units = tune(),
  num_dropout = 1,
  dropout_rate = tune(),
  epochs = 20
) |>
  set_engine("keras")

# 4. Set up a tuning workflow
rec <- recipe(mpg ~ ., data = mtcars) |>
  step_normalize(all_numeric_predictors())

wf_tune <- workflow() |>
  add_recipe(rec) |>
  add_model(tune_spec)

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
  grid = grid
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
