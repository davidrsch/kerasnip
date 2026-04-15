# Fit Method for kerasnip Spec Objects

S3 method for [`fit()`](https://generics.r-lib.org/reference/fit.html)
dispatched on `kerasnip_spec` objects. Delegates to the standard parsnip
[`fit.model_spec()`](https://parsnip.tidymodels.org/reference/fit.html)
and then tags the result with the `kerasnip_model_fit` class so that
[`predict.kerasnip_model_fit()`](https://davidrsch.github.io/kerasnip/dev/reference/predict.kerasnip_model_fit.md)
is dispatched on subsequent calls.

`kerasnip_spec` is stripped from the class before
[`NextMethod()`](https://rdrr.io/r/base/UseMethod.html) to prevent
parsnip's internal `specific_model()` helper from returning more than
one model-class entry, which would break registry lookups. The custom
metadata attributes remain on the object and are thus stored inside the
resulting `model_fit$spec`.

## Usage

``` r
# S3 method for class 'kerasnip_spec'
fit(object, ...)
```

## Arguments

- object:

  A `kerasnip_spec` model specification.

- ...:

  Passed to
  [`parsnip::fit.model_spec()`](https://parsnip.tidymodels.org/reference/fit.html).

## Value

A `model_fit` object with the additional `kerasnip_model_fit` class
prepended to its class vector.
