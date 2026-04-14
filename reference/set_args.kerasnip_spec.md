# set_args Method for kerasnip Spec Objects

S3 method for
[`set_args()`](https://parsnip.tidymodels.org/reference/set_args.html)
dispatched on `kerasnip_spec` objects. `parsnip::set_args.model_spec()`
calls
[`new_model_spec()`](https://parsnip.tidymodels.org/reference/add_on_exports.html),
which strips any extra classes and attributes. This wrapper saves and
re-attaches the `kerasnip_layer_blocks` and `kerasnip_functional`
metadata attributes (and the `kerasnip_spec` class) after
[`NextMethod()`](https://rdrr.io/r/base/UseMethod.html) has done its
work.

## Usage

``` r
# S3 method for class 'kerasnip_spec'
set_args(object, ...)
```

## Arguments

- object:

  A `kerasnip_spec` model specification.

- ...:

  Named model arguments to update, passed to
  `parsnip::set_args.model_spec()`.

## Value

A `model_spec` object with the `kerasnip_spec` class and metadata
attributes re-attached.
