# Register a Custom Keras Loss

Allows users to register a custom loss function so it can be used by
name within `kerasnip` model specifications and tuned with `dials`.

## Usage

``` r
register_keras_loss(name, loss_fn)
```

## Arguments

- name:

  The name to register the loss under (character).

- loss_fn:

  The loss function.

## Value

No return value, called for side effects.

## Details

Registered losses are stored in an internal environment. When a model is
compiled, `kerasnip` will first check this internal registry for a loss
matching the provided name before checking the `keras3` package.

## See also

[`register_keras_optimizer()`](https://davidrsch.github.io/kerasnip/reference/register_keras_optimizer.md),
[`register_keras_metric()`](https://davidrsch.github.io/kerasnip/reference/register_keras_metric.md)
