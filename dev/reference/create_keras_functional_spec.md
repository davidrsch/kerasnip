# Create a Custom Keras Functional API Model Specification for Tidymodels

This function acts as a factory to generate a new `parsnip` model
specification based on user-defined blocks of Keras layers using the
Functional API. This allows for creating complex, tunable architectures
with non-linear topologies that integrate seamlessly with the
`tidymodels` ecosystem.

## Usage

``` r
create_keras_functional_spec(
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
  function (e.g., "custom_resnet"). This should be a valid R function
  name.

- layer_blocks:

  A named list of functions where each function defines a "block" (a
  node) in the model graph. The list names are crucial as they define
  the names of the nodes. The arguments of each function define how the
  nodes are connected. See the "Model Graph Connectivity" section for
  details.

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
model specification function (e.g., `custom_resnet()`) in the specified
environment and register the model with `parsnip` so it can be used
within the `tidymodels` framework.

## Details

This function generates all the boilerplate needed to create a custom,
tunable `parsnip` model specification that uses the Keras Functional
API. This is ideal for models with complex, non-linear topologies, such
as networks with multiple inputs/outputs or residual connections.

The function inspects the arguments of your `layer_blocks` functions and
makes them available as tunable parameters in the generated model
specification, prefixed with the block's name (e.g., `dense_units`).
Common training parameters such as `epochs` and `learn_rate` are also
added.

## Model Graph Connectivity

`kerasnip` builds the model's directed acyclic graph by inspecting the
arguments of each function in the `layer_blocks` list. The connection
logic is as follows:

1.  The **names of the elements** in the `layer_blocks` list define the
    names of the nodes in your graph (e.g., `main_input`, `dense_path`,
    `output`).

2.  The **names of the arguments** in each block function specify its
    inputs. A block function like
    `my_block <- function(input_a, input_b, ...)` declares that it needs
    input from the nodes named `input_a` and `input_b`. `kerasnip` will
    automatically supply the output tensors from those nodes when
    calling `my_block`.

There are two special requirements:

- **Input Block**: The first block in the list is treated as the input
  node. Its function should not take other blocks as input, but it can
  have an `input_shape` argument, which is supplied automatically during
  fitting.

- **Output Block**: Exactly one block must be named `"output"`. The
  tensor returned by this block is used as the final output of the Keras
  model.

A key feature is the automatic creation of `num_{block_name}` arguments
(e.g., `num_dense_path`). This allows you to control how many times a
block is repeated, making it easy to tune the depth of your network. A
block can only be repeated if it has exactly one input from another
block in the graph.

The new model specification function and its
[`update()`](https://rdrr.io/r/stats/update.html) method are created in
the environment specified by the `env` argument.

## See also

[`remove_keras_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/remove_keras_spec.md),
[`parsnip::new_model_spec()`](https://parsnip.tidymodels.org/reference/add_on_exports.html),
[`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/create_keras_sequential_spec.md)

## Examples

``` r
# \donttest{
if (requireNamespace("keras3", quietly = TRUE)) {
  library(keras3)
  library(parsnip)

  # 1. Define block functions. These are the building blocks of our model.
  # An input block that receives the data's shape automatically.
  input_block <- function(input_shape) layer_input(shape = input_shape)

  # A dense block with a tunable `units` parameter.
  dense_block <- function(tensor, units) {
    tensor |> layer_dense(units = units, activation = "relu")
  }

  # A block that adds two tensors together (for the residual connection).
  add_block <- function(input_a, input_b) layer_add(list(input_a, input_b))

  # An output block for regression.
  output_block_reg <- function(tensor) layer_dense(tensor, units = 1)

  # 2. Create the spec. The `layer_blocks` list defines the graph.
  create_keras_functional_spec(
    model_name = "my_resnet_spec",
    layer_blocks = list(
      # The names of list elements are the node names.
      main_input = input_block,

      # The argument `main_input` connects this block to the input node.
      dense_path = function(main_input, units = 32) dense_block(main_input, units),

      # This block's arguments connect it to the original input AND the dense layer.
      add_residual = function(main_input, dense_path) add_block(main_input, dense_path),

      # This block must be named 'output'. It connects to the residual add layer.
      output = function(add_residual) output_block_reg(add_residual)
    ),
    mode = "regression"
  )

  # 3. Use the newly created specification function!
  # The `dense_path_units` argument was created automatically.
  model_spec <- my_resnet_spec(dense_path_units = 64, epochs = 10)

  # You could also tune the number of dense layers since it has a single input:
  # model_spec <- my_resnet_spec(num_dense_path = 2, dense_path_units = 32)

  print(model_spec)
  remove_keras_spec("my_resnet_spec")
  # tune::tunable(model_spec)
}
#> my resnet spec Model Specification (regression)
#> 
#> Main Arguments:
#>   num_main_input = structure(list(), class = "rlang_zap")
#>   num_dense_path = structure(list(), class = "rlang_zap")
#>   num_add_residual = structure(list(), class = "rlang_zap")
#>   num_output = structure(list(), class = "rlang_zap")
#>   dense_path_units = 64
#>   learn_rate = structure(list(), class = "rlang_zap")
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
#> Removed from parsnip registry objects: my_resnet_spec, my_resnet_spec_args, my_resnet_spec_encoding, my_resnet_spec_fit, my_resnet_spec_modes, my_resnet_spec_pkgs, my_resnet_spec_predict
#> Removed 'my_resnet_spec' from parsnip:::get_model_env()$models
# }
```
