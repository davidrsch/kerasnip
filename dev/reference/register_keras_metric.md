# Register a Custom Keras Metric

Allows users to register a custom metric function so it can be used by
name within `kerasnip` model specifications.

## Usage

``` r
register_keras_metric(name, metric_fn)
```

## Arguments

- name:

  The name to register the metric under (character).

- metric_fn:

  The metric function.

## Value

No return value, called for side effects.

## Details

Registered metrics are stored in an internal environment. When a model
is compiled, `kerasnip` will first check this internal registry for a
metric matching the provided name before checking the `keras3` package.

## See also

[`register_keras_optimizer()`](https://davidrsch.github.io/kerasnip/dev/reference/register_keras_optimizer.md),
[`register_keras_loss()`](https://davidrsch.github.io/kerasnip/dev/reference/register_keras_loss.md)
