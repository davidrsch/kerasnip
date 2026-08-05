# Full Conformal Inference Method for `kerasnip_step_view` Objects

Full (refit-per-candidate) conformal intervals for one forecast step of
a multistep fit. Requires
[`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
and
[`step_sequence()`](https://davidrsch.github.io/kerasnip/reference/step_sequence.md)
to share a single source column (true of every multistep model built
with this package's own examples/vignette); see the design note above
`kerasnip_step_recipe_step()`. Only `control$method = "grid"` is
supported.

## Usage

``` r
# S3 method for class 'kerasnip_step_view'
int_conformal_full(object, train_data, ..., control = NULL)
```

## Arguments

- object:

  A `kerasnip_step_view`.

- train_data:

  The raw training data used to fit `object`'s underlying model.

- ...:

  Not used.

- control:

  A
  [`probably::control_conformal_full()`](https://probably.tidymodels.org/reference/control_conformal_full.html)
  object; defaults to `method = "grid"` if not supplied.

## Value

A `kerasnip_conformal_full_step`/`int_conformal_full` object; call
[`predict()`](https://rdrr.io/r/stats/predict.html) on it to get
intervals for new data (a raw continuation of `train_data`).
