# Full Conformal Inference Method for `kerasnip_output_view` Objects

Full (refit-per-candidate) conformal intervals for one output of a
multi-output fit. This is kerasnip's own implementation (see the design
note above `kerasnip_mold_outcome_names()`), not a reuse of `probably`'s
private internals, since those assume a single-outcome fit throughout.
Only `control$method = "grid"` is supported.

## Usage

``` r
# S3 method for class 'kerasnip_output_view'
int_conformal_full(object, train_data, ..., control = NULL)
```

## Arguments

- object:

  A `kerasnip_output_view` viewing a regression output.

- train_data:

  The training data used to fit `object`'s underlying model.

- ...:

  Not used.

- control:

  A
  [`probably::control_conformal_full()`](https://probably.tidymodels.org/reference/control_conformal_full.html)
  object; defaults to `method = "grid"` if not supplied.

## Value

A `kerasnip_conformal_full`/`int_conformal_full` object; call
[`predict()`](https://rdrr.io/r/stats/predict.html) on it to get
intervals for new data.
