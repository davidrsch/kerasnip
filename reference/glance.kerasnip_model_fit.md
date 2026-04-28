# Glance at a Fitted Kerasnip Model

Returns a one-row tibble of summary statistics from the final training
epoch: every metric the model was compiled with (e.g. `loss`,
`accuracy`).

## Usage

``` r
# S3 method for class 'kerasnip_model_fit'
glance(x, ...)
```

## Arguments

- x:

  A `kerasnip_model_fit` object.

- ...:

  Not used.

## Value

A one-row tibble with one column per compiled metric. Returns an empty
tibble if training history has been stripped (e.g. by butcher).
