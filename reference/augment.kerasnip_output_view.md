# Augment Method for `kerasnip_output_view` Objects

Binds `predict(x, new_data, type = "numeric")`'s `.pred` column with
`new_data`, mirroring `workflows:::augment.workflow()`. Used internally
by
[`int_conformal_split.kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/int_conformal_split.kerasnip_output_view.md),
which needs both the prediction and `new_data`'s truth column in one
data frame.

## Usage

``` r
# S3 method for class 'kerasnip_output_view'
augment(x, new_data, ...)
```

## Arguments

- x:

  A `kerasnip_output_view`.

- new_data:

  A data frame of predictors, including `x$output`'s truth column.

- ...:

  Not used.

## Value

`new_data` with a `.pred` column prepended.
