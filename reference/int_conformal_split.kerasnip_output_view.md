# Split Conformal Inference Method for `kerasnip_output_view` Objects

Calibration-set conformal intervals for one output of a multi-output
fit. Mirrors `probably`'s own (private)
`int_conformal_split.workflow()`, using only
[`hardhat::extract_mold()`](https://hardhat.tidymodels.org/reference/hardhat-extract.html)
and
[`generics::augment()`](https://generics.r-lib.org/reference/augment.html)
(both implemented for this class), rather than `probably`'s unexported
internals.

## Usage

``` r
# S3 method for class 'kerasnip_output_view'
int_conformal_split(object, cal_data, ...)
```

## Arguments

- object:

  A `kerasnip_output_view`.

- cal_data:

  A data frame of calibration predictors and truth.

- ...:

  Not used.

## Value

A `conformal_reg_split`/`int_conformal_split` object;
[`predict()`](https://rdrr.io/r/stats/predict.html) on it (from
`probably`) works unmodified, since it dispatches back to
[`predict.kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/predict.kerasnip_output_view.md).
