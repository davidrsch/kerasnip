# Predict Method for kerasnip Model Fits

S3 method for [`predict()`](https://rdrr.io/r/stats/predict.html)
dispatched on `kerasnip_model_fit` objects. Before delegating to the
standard parsnip predict machinery, it checks whether the underlying
model type is registered in the current parsnip session. If not (e.g.
after loading a saved workflow in a new R session), it transparently
replays the full parsnip registration using metadata stored on the spec
object — requiring no manual step from the user.

## Usage

``` r
# S3 method for class 'kerasnip_model_fit'
predict(object, new_data, ...)
```

## Arguments

- object:

  A `kerasnip_model_fit` object.

- new_data:

  A data frame of predictors.

- ...:

  Passed to the parsnip predict method.

## Value

A tibble of predictions.

## Details

The metadata needed for re-registration (`kerasnip_layer_blocks`,
`kerasnip_functional`) is embedded on the spec object by the spec
constructor function at call time. This means it is preserved across
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html)/[`readRDS()`](https://rdrr.io/r/base/readRDS.html)
and
[`bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)/[`unbundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
round-trips.

For full model weight portability (i.e. to be able to
[`predict()`](https://rdrr.io/r/stats/predict.html) on new data in a new
R session), use
[`bundle::bundle()`](https://rstudio.github.io/bundle/reference/bundle.html)
before saving. Plain [`saveRDS()`](https://rdrr.io/r/base/readRDS.html)
preserves the spec structure and will auto-register, but the underlying
Keras model weights are not portable without bundling.
