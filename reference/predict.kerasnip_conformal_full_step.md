# Predict Method for `kerasnip_conformal_full_step` Objects

Computes full-conformal intervals for `new_data` (a raw continuation of
the training data), one grid search per surviving window via
`kerasnip_grid_one_step_view()`.

## Usage

``` r
# S3 method for class 'kerasnip_conformal_full_step'
predict(object, new_data, level = 0.95, ...)
```

## Arguments

- object:

  A `kerasnip_conformal_full_step` object, from
  [`int_conformal_full.kerasnip_step_view()`](https://davidrsch.github.io/kerasnip/reference/int_conformal_full.kerasnip_step_view.md).

- new_data:

  Raw data continuing the training series.

- level:

  The conformal level.

- ...:

  Not used.

## Value

A tibble with `.pred_lower`/`.pred_upper` columns, one row per window
that survives
[`step_sequence()`](https://davidrsch.github.io/kerasnip/reference/step_sequence.md)'s
history requirement.
