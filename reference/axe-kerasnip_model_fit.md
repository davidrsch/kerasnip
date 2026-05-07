# Butcher axe methods for kerasnip_model_fit

These methods allow
[`butcher::butcher()`](https://butcher.tidymodels.org/reference/butcher.html)
to reduce the memory footprint of fitted kerasnip model objects. The
Keras model itself (stored as raw bytes in `$fit$keras_bytes`) is always
preserved so that [`predict()`](https://rdrr.io/r/stats/predict.html)
continues to work after butchering.

The main saving comes from `axe_data()`, which removes the training
history object (`$fit$history`). For long training runs this can be
several MB.

## Usage

``` r
# S3 method for class 'kerasnip_model_fit'
axe_data(x, verbose = FALSE, ...)

# S3 method for class 'kerasnip_model_fit'
axe_env(x, verbose = FALSE, ...)

# S3 method for class 'kerasnip_model_fit'
axe_call(x, verbose = FALSE, ...)

# S3 method for class 'kerasnip_model_fit'
axe_ctrl(x, verbose = FALSE, ...)

# S3 method for class 'kerasnip_model_fit'
axe_fitted(x, verbose = FALSE, ...)
```

## Arguments

- x:

  A `kerasnip_model_fit` object.

- verbose:

  Logical. Print information about memory released and disabled
  functions. Default is `FALSE`.

- ...:

  Not used.

## Value

An axed `kerasnip_model_fit` object with the
`butcher_kerasnip_model_fit` class prepended.
