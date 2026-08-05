# View a Single Forecast Step of a Multistep kerasnip Fit

A kerasnip multistep (vector-valued) regression model returns a nested
`.pred` list-column: one inner tibble per row, with a `.step` column
plus one prediction column per forecasted variable. `tailor`/`probably`
expect a single flat numeric `.pred` column instead —
`tailor::check_variable_type()` requires
[`is.numeric()`](https://rdrr.io/r/base/numeric.html) on the
outcome/estimate columns, which a list-column fails outright.

`kerasnip_step_view()` wraps a fitted multistep workflow together with
one forecast step (and, if more than one variable is forecast, which
variable), presenting it as an ordinary single-output fit:
[`predict()`](https://rdrr.io/r/stats/predict.html) returns a flat
`.pred` column for that step alone.

## Usage

``` r
kerasnip_step_view(x, step, var = NULL)
```

## Arguments

- x:

  A fitted (trained) `workflow` whose model is a multistep regression
  model (see
  [`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_sequential_spec.md)/
  [`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_functional_spec.md)
  with a vector-valued output).

- step:

  An integer, the forecast step to view.

- var:

  A string, the forecasted variable to view. Required only if the model
  forecasts more than one variable; inferred otherwise.

## Value

A `kerasnip_step_view` object.

## Details

Unlike
[`kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_output_view.md),
a multistep model's per-step outcome columns (e.g. `lead_1_value`) are
recipe-*engineered* from a single raw column via
[`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
— they are not present in a user's raw data the way genuine multi-output
columns are.
[`kerasnip_step_truth()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_step_truth.md)
recovers the true future value at a given step by re-baking the fitted
recipe on raw data, which is what
[int_conformal_split()](https://probably.tidymodels.org/reference/int_conformal_split.html)
uses internally for this class.

[`probably::int_conformal_full()`](https://probably.tidymodels.org/reference/int_conformal_full.html)
is also supported (see
[`int_conformal_full.kerasnip_step_view()`](https://davidrsch.github.io/kerasnip/reference/int_conformal_full.kerasnip_step_view.md)),
with a materially different design from
[`kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_output_view.md)'s:
refitting for a candidate value at this step means substituting it into
the single *raw* column
[`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
derives every step's truth from, which shifts every nearby row's target
too. It is only supported when
[`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
and
[`step_sequence()`](https://davidrsch.github.io/kerasnip/reference/step_sequence.md)
share a single source column, matching all of kerasnip's own multistep
examples.

## Examples

``` r
if (FALSE) { # \dontrun{
fit_obj <- fit(wf, data = train_data) # a multistep forecasting workflow
step_2 <- kerasnip_step_view(fit_obj, step = 2)
predict(step_2, new_data = test_data) # -> a single `.pred` column
} # }
```
