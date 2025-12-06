# Register a Custom Keras Optimizer

Allows users to register a custom optimizer function so it can be used
by name within `kerasnip` model specifications and tuned with `dials`.

## Usage

``` r
register_keras_optimizer(name, optimizer_fn)
```

## Arguments

- name:

  The name to register the optimizer under (character).

- optimizer_fn:

  The optimizer function. It should return a Keras optimizer object.

## Value

No return value, called for side effects.

## Details

Registered optimizers are stored in an internal environment. When a
model is compiled, `kerasnip` will first check this internal registry
for an optimizer matching the provided name before checking the `keras3`
package.

The `optimizer_fn` can be a simple function or a partially applied
function using
[`purrr::partial()`](https://purrr.tidyverse.org/reference/partial.html).
This is useful for creating versions of Keras optimizers with specific
settings.

## See also

[`register_keras_loss()`](https://davidrsch.github.io/kerasnip/reference/register_keras_loss.md),
[`register_keras_metric()`](https://davidrsch.github.io/kerasnip/reference/register_keras_metric.md)

## Examples

``` r
if (requireNamespace("keras3", quietly = TRUE)) {
  # Register a custom version of Adam with a different default beta_1
  my_adam <- purrr::partial(keras3::optimizer_adam, beta_1 = 0.8)
  register_keras_optimizer("my_adam", my_adam)

  # Now "my_adam" can be used as a string in a model spec, e.g.,
  # my_model_spec(compile_optimizer = "my_adam")
}
```
