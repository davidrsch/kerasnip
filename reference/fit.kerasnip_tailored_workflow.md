# Fit Method for `kerasnip_tailored_workflow` Objects

Trains the underlying multi-output/multistep workflow, then fits
`object$tailor` against the target output's/step's predictions (via
[`kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_output_view.md)/[`kerasnip_step_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_step_view.md))
on `data_calibration` if supplied, otherwise on `data`.

## Usage

``` r
# S3 method for class 'kerasnip_tailored_workflow'
fit(object, data, ..., data_calibration = NULL)
```

## Arguments

- object:

  A `kerasnip_tailored_workflow`, from
  [`kerasnip_add_tailor()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_add_tailor.md).

- data:

  The training data.

- ...:

  Passed to [`fit()`](https://generics.r-lib.org/reference/fit.html) on
  the underlying workflow.

- data_calibration:

  Optional calibration data for the `tailor`; defaults to `data` if not
  supplied.

## Value

A `kerasnip_tailored_fit`, to be used with
[`predict()`](https://rdrr.io/r/stats/predict.html).
