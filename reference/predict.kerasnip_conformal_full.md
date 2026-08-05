# Predict Method for `kerasnip_conformal_full` Objects

Computes full-conformal intervals for `new_data`, one grid search per
row via `kerasnip_grid_one_output_view()`.

## Usage

``` r
# S3 method for class 'kerasnip_conformal_full'
predict(object, new_data, level = 0.95, ...)
```

## Arguments

- object:

  A `kerasnip_conformal_full` object, from
  [`int_conformal_full.kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/int_conformal_full.kerasnip_output_view.md).

- new_data:

  A data frame of predictors.

- level:

  The conformal level.

- ...:

  Not used.

## Value

A tibble with `.pred_lower`/`.pred_upper` columns, one row per row of
`new_data`.
