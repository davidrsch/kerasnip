# Attach a `tailor` Post-Processor to One Output or Step of a Multi-Output or Multistep Workflow

[`workflows::add_tailor()`](https://workflows.tidymodels.org/reference/add_tailor.html)
cannot be used on a kerasnip multi-output or multistep workflow:
[`tailor::fit()`](https://generics.r-lib.org/reference/fit.html) selects
`outcome`/`estimate` via `[[`, which requires exactly one, flat, numeric
column, and both a multi-output recipe (`output_1 + output_2 ~ .`) and a
multistep model's nested `.pred` list-column violate that (see
[`vignette("multi_output_postprocessing")`](https://davidrsch.github.io/kerasnip/articles/multi_output_postprocessing.md)).
`kerasnip_add_tailor()` is a kerasnip-owned analogue that attaches a
`tailor` post-processor to a single named output or forecast step, using
[`kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_output_view.md)
or
[`kerasnip_step_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_step_view.md)
internally.

## Usage

``` r
kerasnip_add_tailor(x, tailor, output = NULL, step = NULL, var = NULL)
```

## Arguments

- x:

  An **unfitted** `workflow` whose model has more than one outcome
  (multi-output) or is a multistep forecasting model.

- tailor:

  A
  [`tailor::tailor()`](https://tailor.tidymodels.org/reference/tailor.html)
  specification.

- output:

  A string, the name of the outcome column to post-process (multi-output
  models).

- step:

  An integer, the forecast step to post-process (multistep models).

- var:

  A string, the forecasted variable to post-process; only needed with
  `step` if the model forecasts more than one variable.

## Value

A `kerasnip_tailored_workflow`, to be trained with
[`fit()`](https://generics.r-lib.org/reference/fit.html).

## Details

At [`fit()`](https://generics.r-lib.org/reference/fit.html) time, the
underlying model is trained as usual; the relevant view is then used to
fit the `tailor` against that output's/step's predictions (on
`data_calibration` if supplied, otherwise on `data`, mirroring
[`workflows::add_tailor()`](https://workflows.tidymodels.org/reference/add_tailor.html)'s
data-usage convention). At
[`predict()`](https://rdrr.io/r/stats/predict.html) time, the full
prediction is generated, the target output's/step's value(s) are
replaced with the tailor-adjusted values, and everything else (other
outputs; other steps in the same nested tibble) is left untouched.

Exactly one of `output` or `step` must be supplied: `output` for a
multi-output model, `step` (and `var`, if more than one variable is
forecast) for a multistep model.

## Examples

``` r
if (FALSE) { # \dontrun{
tlr <- tailor::tailor() |> tailor::adjust_numeric_calibration()

# multi-output
tailored_wf <- kerasnip_add_tailor(wf, tlr, output = "output_1")

# multistep
tailored_wf <- kerasnip_add_tailor(wf, tlr, step = 2)

fit_obj <- fit(tailored_wf, data = train_data, data_calibration = cal_data)
predict(fit_obj, new_data = test_data)
} # }
```
