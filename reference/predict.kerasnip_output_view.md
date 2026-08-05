# Predict Method for `kerasnip_output_view` Objects

Predicts from the wrapped multi-output workflow, then selects and
renames `object$output`'s columns down to the standard single-output
shape (`.pred`, `.pred_class`, `.pred_<level>`, or
`.pred`/`.pred_lower`/ `.pred_upper`), so the result reads like a
single-output [`predict()`](https://rdrr.io/r/stats/predict.html) call.

## Usage

``` r
# S3 method for class 'kerasnip_output_view'
predict(object, new_data, type = NULL, ...)
```

## Arguments

- object:

  A `kerasnip_output_view`.

- new_data:

  A data frame of predictors.

- type:

  One of `"numeric"`, `"class"`, `"prob"`, `"conf_int"`, or
  `"pred_int"`. Defaults to `"class"` for a classification view,
  ` "numeric"` otherwise.

- ...:

  Passed to [`predict()`](https://rdrr.io/r/stats/predict.html) on the
  wrapped workflow.

## Value

A tibble in the standard single-output prediction shape.
