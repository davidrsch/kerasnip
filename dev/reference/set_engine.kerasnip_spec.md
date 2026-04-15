# set_engine Method for kerasnip Spec Objects

S3 method for
[`set_engine()`](https://parsnip.tidymodels.org/reference/set_engine.html)
dispatched on `kerasnip_spec` objects.
`parsnip::set_engine.model_spec()` internally calls
[`new_model_spec()`](https://parsnip.tidymodels.org/reference/add_on_exports.html),
which re-creates the spec from scratch with only `c(cls, "model_spec")`
as the class vector — stripping `kerasnip_spec` and any custom
attributes. This wrapper preserves the `kerasnip_layer_blocks` and
`kerasnip_functional` metadata attributes and re-attaches them (along
with the `kerasnip_spec` class) after
[`NextMethod()`](https://rdrr.io/r/base/UseMethod.html) has done its
work.

## Usage

``` r
# S3 method for class 'kerasnip_spec'
set_engine(object, engine, ...)
```

## Arguments

- object:

  A `kerasnip_spec` model specification.

- engine:

  A character string naming the engine (e.g., `"keras"`).

- ...:

  Additional engine-specific arguments passed to
  `parsnip::set_engine.model_spec()`.

## Value

A `model_spec` object with the `kerasnip_spec` class and metadata
attributes re-attached.
