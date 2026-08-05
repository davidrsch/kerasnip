# Predict Method for `kerasnip_tailored_fit` Objects

Predicts from the underlying full workflow, applies the fitted `tailor`
to the target output's/step's predictions, and splices the adjusted
values back in: `.pred_<output>`/`.pred_class_<output>`-suffixed columns
for a multi-output model, or the matching `.step` entry in every row's
nested tibble for a multistep model. Every other output/step is returned
exactly as a plain [`predict()`](https://rdrr.io/r/stats/predict.html)
on the underlying fit would give it.

## Usage

``` r
# S3 method for class 'kerasnip_tailored_fit'
predict(object, new_data, ...)
```

## Arguments

- object:

  A `kerasnip_tailored_fit`, from
  [`fit()`](https://generics.r-lib.org/reference/fit.html) on a
  `kerasnip_tailored_workflow`.

- new_data:

  A data frame of predictors.

- ...:

  Not used.

## Value

A tibble in the full multi-output/multistep prediction shape.
