# Split Conformal Inference Method for `kerasnip_step_view` Objects

Calibration-set conformal intervals for one forecast step of a multistep
fit. Mirrors `probably`'s own (private)
`int_conformal_split.workflow()`, using only
[`generics::augment()`](https://generics.r-lib.org/reference/augment.html)
(implemented for this class via
[`kerasnip_step_truth()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_step_truth.md)),
rather than `probably`'s unexported internals.

## Usage

``` r
# S3 method for class 'kerasnip_step_view'
int_conformal_split(object, cal_data, ...)
```

## Arguments

- object:

  A `kerasnip_step_view`.

- cal_data:

  A data frame of raw calibration predictors (and the original outcome
  column
  [`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
  was applied to).

- ...:

  Not used.

## Value

A `conformal_reg_split`/`int_conformal_split` object;
[`predict()`](https://rdrr.io/r/stats/predict.html) on it (from
`probably`) works unmodified, since it dispatches back to
[`predict.kerasnip_step_view()`](https://davidrsch.github.io/kerasnip/reference/predict.kerasnip_step_view.md).
