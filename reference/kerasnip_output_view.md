# View a Single Output of a Multi-Output kerasnip Fit

`tailor` and `probably` are built around models with a single outcome
column and a single `.pred`/`.pred_class` prediction column. A kerasnip
multi-output model (e.g. a recipe with `output_1 + output_2 ~ .`)
instead produces `.pred_output_1`, `.pred_output_2`, ... columns from
multiple truth columns in one fit — the standard
`parsnip::maybe_multivariate()` shape, but one
[`tailor::fit()`](https://generics.r-lib.org/reference/fit.html)/[`workflows::add_tailor()`](https://workflows.tidymodels.org/reference/add_tailor.html)
call cannot consume it (it selects `outcome`/`estimate` via `[[`, which
requires exactly one column).

`kerasnip_output_view()` wraps a fitted multi-output workflow together
with one output name, presenting it as if it were an ordinary
single-output fit: [`predict()`](https://rdrr.io/r/stats/predict.html)
returns standard `.pred` / `.pred_class` / `.pred_<level>` columns for
that output alone, letting you calibrate or post-process each output
separately with the usual `tailor`/`probably` calls (see
[`vignette("multi_output_postprocessing")`](https://davidrsch.github.io/kerasnip/articles/multi_output_postprocessing.md)).

## Usage

``` r
kerasnip_output_view(x, output)
```

## Arguments

- x:

  A fitted (trained) `workflow` whose model has more than one outcome
  column.

- output:

  A string, the name of the outcome column to view.

## Value

A `kerasnip_output_view` object.

## Examples

``` r
if (FALSE) { # \dontrun{
fit_obj <- fit(wf, data = train_data) # wf predicts output_1 and output_2
view_1 <- kerasnip_output_view(fit_obj, "output_1")
predict(view_1, new_data = test_data) # -> a single `.pred` column
} # }
```
