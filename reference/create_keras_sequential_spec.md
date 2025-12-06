# Create a Custom Keras Sequential Model Specification for Tidymodels

This function acts as a factory to generate a new `parsnip` model
specification based on user-defined blocks of Keras layers using the
Sequential API. This is the ideal choice for creating models that are a
simple, linear stack of layers. For models with complex, non-linear
topologies, see
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_functional_spec.md).

## Usage

``` r
create_keras_sequential_spec(
  model_name,
  layer_blocks,
  mode = c("regression", "classification"),
  ...,
  env = parent.frame()
)
```

## Arguments

- model_name:

  A character string for the name of the new model specification
  function (e.g., "custom_cnn"). This should be a valid R function name.

- layer_blocks:

  A named, ordered list of functions. Each function defines a "block" of
  Keras layers. The function must take a Keras model object as its first
  argument and return the modified model. Other arguments to the
  function will become tunable parameters in the final model
  specification.

- mode:

  A character string, either "regression" or "classification".

- ...:

  Reserved for future use. Currently not used.

- env:

  The environment in which to create the new model specification
  function and its associated
  [`update()`](https://rdrr.io/r/stats/update.html) method. Defaults to
  the calling environment
  ([`parent.frame()`](https://rdrr.io/r/base/sys.parent.html)).

## Value

Invisibly returns `NULL`. Its primary side effect is to create a new
model specification function (e.g., `my_mlp()`) in the specified
environment and register the model with `parsnip` so it can be used
within the `tidymodels` framework.

## Details

This function generates all the boilerplate needed to create a custom,
tunable `parsnip` model specification that uses the Keras Sequential
API.

The function inspects the arguments of your `layer_blocks` functions
(ignoring special arguments like `input_shape` and `num_classes`) and
makes them available as arguments in the generated model specification,
prefixed with the block's name (e.g., `dense_units`).

The new model specification function and its
[`update()`](https://rdrr.io/r/stats/update.html) method are created in
the environment specified by the `env` argument.

## Model Architecture (Sequential API)

`kerasnip` builds the model by applying the functions in `layer_blocks`
in the order they are provided. Each function receives the Keras model
built by the previous function and returns a modified version.

1.  The **first block** must initialize the model (e.g., with
    [`keras_model_sequential()`](https://keras3.posit.co/reference/keras_model_sequential.html)).
    It can accept an `input_shape` argument, which `kerasnip` will
    provide automatically during fitting.

2.  **Subsequent blocks** add layers to the model.

3.  The **final block** should add the output layer. For classification,
    it can accept a `num_classes` argument, which is provided
    automatically.

A key feature of this function is the automatic creation of
`num_{block_name}` arguments (e.g., `num_hidden`). This allows you to
control how many times each block is repeated, making it easy to tune
the depth of your network.

## See also

[`remove_keras_spec()`](https://davidrsch.github.io/kerasnip/reference/remove_keras_spec.md),
[`parsnip::new_model_spec()`](https://parsnip.tidymodels.org/reference/add_on_exports.html),
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_functional_spec.md)

## Examples

``` r
# \donttest{
if (requireNamespace("keras3", quietly = TRUE)) {
library(keras3)
library(parsnip)
library(dials)

# 1. Define layer blocks for a complete model.
# The first block must initialize the model. `input_shape` is passed automatically.
input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}
# A block for hidden layers. `units` will become a tunable parameter.
hidden_block <- function(model, units = 32) {
  model |> layer_dense(units = units, activation = "relu")
}

# The output block. `num_classes` is passed automatically for classification.
output_block <- function(model, num_classes) {
  model |> layer_dense(units = num_classes, activation = "softmax")
}

# 2. Create the spec, providing blocks in the correct order.
create_keras_sequential_spec(
model_name = "my_mlp_seq_spec",
  layer_blocks = list(
    input = input_block,
    hidden = hidden_block,
    output = output_block
  ),
  mode = "classification"
)

# 3. Use the newly created specification function!
# Note the new arguments `num_hidden` and `hidden_units`.
model_spec <- my_mlp_seq_spec(
  num_hidden = 2,
  hidden_units = 64,
  epochs = 10,
  learn_rate = 0.01
)

print(model_spec)
remove_keras_spec("my_mlp_seq_spec")
}
#> my mlp seq spec Model Specification (classification)
#> 
#> Main Arguments:
#>   num_input = structure(list(), class = "rlang_zap")
#>   num_hidden = 2
#>   num_output = structure(list(), class = "rlang_zap")
#>   hidden_units = 64
#>   learn_rate = 0.01
#>   fit_batch_size = structure(list(), class = "rlang_zap")
#>   fit_epochs = structure(list(), class = "rlang_zap")
#>   fit_callbacks = structure(list(), class = "rlang_zap")
#>   fit_validation_split = structure(list(), class = "rlang_zap")
#>   fit_validation_data = structure(list(), class = "rlang_zap")
#>   fit_shuffle = structure(list(), class = "rlang_zap")
#>   fit_class_weight = structure(list(), class = "rlang_zap")
#>   fit_sample_weight = structure(list(), class = "rlang_zap")
#>   fit_initial_epoch = structure(list(), class = "rlang_zap")
#>   fit_steps_per_epoch = structure(list(), class = "rlang_zap")
#>   fit_validation_steps = structure(list(), class = "rlang_zap")
#>   fit_validation_batch_size = structure(list(), class = "rlang_zap")
#>   fit_validation_freq = structure(list(), class = "rlang_zap")
#>   fit_verbose = structure(list(), class = "rlang_zap")
#>   fit_view_metrics = structure(list(), class = "rlang_zap")
#>   compile_optimizer = structure(list(), class = "rlang_zap")
#>   compile_loss = structure(list(), class = "rlang_zap")
#>   compile_metrics = structure(list(), class = "rlang_zap")
#>   compile_loss_weights = structure(list(), class = "rlang_zap")
#>   compile_weighted_metrics = structure(list(), class = "rlang_zap")
#>   compile_run_eagerly = structure(list(), class = "rlang_zap")
#>   compile_steps_per_execution = structure(list(), class = "rlang_zap")
#>   compile_jit_compile = structure(list(), class = "rlang_zap")
#>   compile_auto_scale_loss = structure(list(), class = "rlang_zap")
#>   epochs = 10
#> 
#> Removed from parsnip registry objects: my_mlp_seq_spec, my_mlp_seq_spec_args, my_mlp_seq_spec_encoding, my_mlp_seq_spec_fit, my_mlp_seq_spec_modes, my_mlp_seq_spec_pkgs, my_mlp_seq_spec_predict
#> Removed 'my_mlp_seq_spec' from parsnip:::get_model_env()$models
# }
```
