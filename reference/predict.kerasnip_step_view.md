# Predict Method for `kerasnip_step_view` Objects

Predicts from the wrapped multistep workflow, then extracts
`object$step`'s (and, if set, `object$var`'s) value from every row's
nested `.pred` tibble into a flat column, so the result reads like a
single-output [`predict()`](https://rdrr.io/r/stats/predict.html) call.

## Usage

``` r
# S3 method for class 'kerasnip_step_view'
predict(object, new_data, type = "numeric", ...)
```

## Arguments

- object:

  A `kerasnip_step_view`.

- new_data:

  A data frame of predictors.

- type:

  One of `"numeric"`, `"conf_int"`, or `"pred_int"`.

- ...:

  Passed to [`predict()`](https://rdrr.io/r/stats/predict.html) on the
  wrapped workflow.

## Value

A tibble with a `.pred` column (`"numeric"`), or `.pred`/
`.pred_lower`/`.pred_upper` (`"conf_int"`/`"pred_int"`).
