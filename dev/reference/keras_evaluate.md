# Evaluate a Kerasnip Model

This function provides an `kera_evaluate()` method for `model_fit`
objects created by `kerasnip`. It preprocesses the new data into the
format expected by Keras and then calls
[`keras3::evaluate()`](https://rdrr.io/pkg/tensorflow/man/evaluate.html)
on the underlying model to compute the loss and any other metrics.

## Usage

``` r
keras_evaluate(object, x, y = NULL, ...)
```

## Arguments

- object:

  A `model_fit` object produced by a `kerasnip` specification.

- x:

  A data frame or matrix of new predictor data.

- y:

  A vector or data frame of new outcome data corresponding to `x`.

- ...:

  Additional arguments passed on to
  [`keras3::evaluate()`](https://rdrr.io/pkg/tensorflow/man/evaluate.html)
  (e.g., `batch_size`).

## Value

A named list containing the evaluation results (e.g., `loss`,
`accuracy`). The names are determined by the metrics the model was
compiled with.

## Details

Evaluate a Fitted Kerasnip Model on New Data

## Examples

``` r
# \donttest{
if (requireNamespace("keras3", quietly = TRUE)) {
library(keras3)
library(parsnip)

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

# 2. Define and fit a model ----
create_keras_sequential_spec(
  model_name = "my_mlp_tools",
  layer_blocks = list(
    input = input_block,
    hidden = hidden_block,
    output = output_block
  ),
  mode = "classification"
)

mlp_spec <- my_mlp_tools(
  hidden_units = 32,
  compile_loss = "categorical_crossentropy",
  compile_optimizer = "adam",
  compile_metrics = "accuracy",
  fit_epochs = 5
) |> set_engine("keras")

x_train <- matrix(rnorm(100 * 10), ncol = 10)
y_train <- factor(sample(0:1, 100, replace = TRUE))
train_df <- data.frame(x = I(x_train), y = y_train)

fitted_mlp <- fit(mlp_spec, y ~ x, data = train_df)

# 3. Evaluate the model on new data ----
x_test <- matrix(rnorm(50 * 10), ncol = 10)
y_test <- factor(sample(0:1, 50, replace = TRUE))

eval_metrics <- keras_evaluate(fitted_mlp, x_test, y_test)
print(eval_metrics)

# 4. Extract the Keras model object ----
keras_model <- extract_keras_model(fitted_mlp)
summary(keras_model)

# 5. Extract the training history ----
history <- extract_keras_history(fitted_mlp)
plot(history)
remove_keras_spec("my_mlp_tools")
}
#> $accuracy
#> [1] 0.56
#> 
#> $loss
#> [1] 0.7778158
#> 
#> Model: "sequential_9"
#> ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
#> ┃ Layer (type)                      ┃ Output Shape             ┃       Param # ┃
#> ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
#> │ dense_15 (Dense)                  │ (None, 32)               │           352 │
#> ├───────────────────────────────────┼──────────────────────────┼───────────────┤
#> │ dense_16 (Dense)                  │ (None, 2)                │            66 │
#> └───────────────────────────────────┴──────────────────────────┴───────────────┘
#>  Total params: 1,256 (4.91 KB)
#>  Trainable params: 418 (1.63 KB)
#>  Non-trainable params: 0 (0.00 B)
#>  Optimizer params: 838 (3.28 KB)
#> Removed from parsnip registry objects: my_mlp_tools, my_mlp_tools_args, my_mlp_tools_encoding, my_mlp_tools_fit, my_mlp_tools_modes, my_mlp_tools_pkgs, my_mlp_tools_predict
#> Removed 'my_mlp_tools' from parsnip:::get_model_env()$models
# }
```
