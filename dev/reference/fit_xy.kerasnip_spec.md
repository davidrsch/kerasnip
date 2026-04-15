# fit_xy Method for kerasnip Spec Objects

S3 method for
[`fit_xy()`](https://generics.r-lib.org/reference/fit_xy.html)
dispatched on `kerasnip_spec` objects. Workflows route through `fit_xy`
rather than `fit`, so this method ensures the `kerasnip_model_fit` class
is attached in the workflow fitting path as well.

`kerasnip_spec` is stripped from the class before
[`NextMethod()`](https://rdrr.io/r/base/UseMethod.html) to prevent
parsnip's internal `specific_model()` helper from returning more than
one model-class entry, which would break registry lookups. The custom
metadata attributes remain on the object and are thus stored inside the
resulting `model_fit$spec`.

## Usage

``` r
# S3 method for class 'kerasnip_spec'
fit_xy(object, ...)
```

## Arguments

- object:

  A `kerasnip_spec` model specification.

- ...:

  Passed to
  [`parsnip::fit_xy.model_spec()`](https://parsnip.tidymodels.org/reference/fit.html).

## Value

A `model_fit` object with the additional `kerasnip_model_fit` class
prepended to its class vector.
