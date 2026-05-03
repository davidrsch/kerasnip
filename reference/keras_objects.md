# Dynamically Discovered Keras Objects

These exported vectors contain the names of optimizers, losses, and
metrics discovered from the installed `keras3` package when `kerasnip`
is loaded. This ensures that `kerasnip` is always up-to-date with your
Keras version.

## Usage

``` r
keras_optimizers

keras_losses

keras_metrics
```

## Format

An object of class `character` of length 12.

An object of class `character` of length 21.

An object of class `character` of length 32.

## Details

These objects are primarily used to provide the default `values` for the
`dials` parameter functions,
[`optimizer_function()`](https://davidrsch.github.io/kerasnip/reference/optimizer_function.md)
and
[`loss_function_keras()`](https://davidrsch.github.io/kerasnip/reference/loss_function_keras.md).
This allows for tab-completion in IDEs and validation of optimizer and
loss names when tuning models.

The discovery process in `.onLoad()` scrapes the `keras3` namespace for
functions matching `optimizer_*`, `loss_*`, and `metric_*` patterns.
