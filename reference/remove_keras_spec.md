# Remove a Keras Model Specification and its Registrations

This function completely removes a model specification that was
previously created by
[`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_sequential_spec.md)
or
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_functional_spec.md).
It cleans up both the function in the user's environment and all
associated registrations within the `parsnip` package.

## Usage

``` r
remove_keras_spec(model_name, env = parent.frame())
```

## Arguments

- model_name:

  A character string giving the name of the model specification function
  to remove (e.g., "my_mlp").

- env:

  The environment from which to remove the function and its
  [`update()`](https://rdrr.io/r/stats/update.html) method. Defaults to
  the calling environment
  ([`parent.frame()`](https://rdrr.io/r/base/sys.parent.html)).

## Value

Invisibly returns `TRUE` after attempting to remove the objects.

## Details

This function is essential for cleanly unloading a dynamically created
model. It performs three main actions:

1.  It removes the model specification function (e.g., `my_mlp()`) and
    its corresponding [`update()`](https://rdrr.io/r/stats/update.html)
    method from the specified environment.

2.  It searches `parsnip`'s internal model environment for all objects
    whose names start with the `model_name` and removes them. This
    purges the fit methods, argument definitions, and other
    registrations.

3.  It removes the model's name from `parsnip`'s master list of models.

This function uses the un-exported
[`get_model_env()`](https://davidrsch.github.io/kerasnip/reference/get_model_env.md)
to perform the cleanup.

## See also

[`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_sequential_spec.md),
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_functional_spec.md)

## Examples

``` r
# \donttest{
if (requireNamespace("keras3", quietly = TRUE)) {
  # First, create a dummy spec
  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units)
  }
  create_keras_sequential_spec(
    "my_temp_model",
    list(
      input = input_block,
      dense = dense_block
    ),
    "regression"
  )

  # Check it exists in the environment and in parsnip
  exists("my_temp_model")
  "my_temp_model" %in% parsnip::show_engines("my_temp_model")$model

  # Now remove it
  remove_keras_spec("my_temp_model")

  # Check it's gone
  !exists("my_temp_model")
  !model_exists("my_temp_model")
}
#> Warning: Unknown or uninitialised column: `model`.
#> Removed from parsnip registry objects: my_temp_model, my_temp_model_args, my_temp_model_encoding, my_temp_model_fit, my_temp_model_modes, my_temp_model_pkgs, my_temp_model_predict
#> Removed 'my_temp_model' from parsnip:::get_model_env()$models
#> [1] TRUE
# }
```
