# Tidy a Fitted Kerasnip Model

Returns a tibble with one row per layer of the underlying Keras model,
summarising the layer name, Python class, and parameter count.

## Usage

``` r
# S3 method for class 'kerasnip_model_fit'
tidy(x, ...)
```

## Arguments

- x:

  A `kerasnip_model_fit` object.

- ...:

  Not used.

## Value

A tibble with columns `layer` (character), `class` (character), and
`n_params` (integer).
