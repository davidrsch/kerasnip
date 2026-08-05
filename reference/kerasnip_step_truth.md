# Recover Truth Values for a Multistep Forecast Step

A multistep model's per-step outcome columns (e.g. `lead_2_value`) are
engineered by
[`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
from a single raw column, so they are not present in a user's raw data
the way genuine multi-output columns are. This re-bakes the fitted
recipe on `new_data` to recover the actual future value at
[`kerasnip_step_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_step_view.md)'s
step, for calibration/interval use. Rows too close to the end of
`new_data` for the lead to be computed return `NA` (dropped
automatically by calibration routines that call
[`sort()`](https://rdrr.io/r/base/sort.html)/[`stats::complete.cases()`](https://rdrr.io/r/stats/complete.cases.html)
on the result).

## Usage

``` r
kerasnip_step_truth(view, new_data)
```

## Arguments

- view:

  A `kerasnip_step_view`.

- new_data:

  A data frame of raw predictors (and the original outcome column
  [`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
  was applied to).

## Value

A numeric vector, one truth value per row of `new_data`.
