# Augment Method for `kerasnip_step_view` Objects

Binds `predict(x, new_data, type = "numeric")`'s `.pred` column with the
step's truth (from
[`kerasnip_step_truth()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_step_truth.md))
and `new_data`, mirroring `workflows:::augment.workflow()`. Used
internally by
[`int_conformal_split.kerasnip_step_view()`](https://davidrsch.github.io/kerasnip/reference/int_conformal_split.kerasnip_step_view.md).
Rows
[`step_sequence()`](https://davidrsch.github.io/kerasnip/reference/step_sequence.md)
drops for lacking a full window of history are dropped here too, to stay
aligned with [`predict()`](https://rdrr.io/r/stats/predict.html)'s row
count.

## Usage

``` r
# S3 method for class 'kerasnip_step_view'
augment(x, new_data, ...)
```

## Arguments

- x:

  A `kerasnip_step_view`.

- new_data:

  A data frame of raw predictors (and the original outcome column
  [`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
  was applied to).

- ...:

  Not used.

## Value

A tibble: `.pred`, the step's truth column (named `x$outcome_col`), and
`new_data`'s columns, aligned to the rows that survived windowing.
