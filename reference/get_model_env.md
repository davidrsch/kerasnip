# Get Parsnip's Model Environment

This is an internal helper function to retrieve the environment where
`parsnip` stores its model definitions. It is used to dynamically
interact with the `parsnip` infrastructure.

## Usage

``` r
get_model_env()
```

## Value

The `parsnip` model environment.

## Examples

``` r
# \donttest{
model_env <- kerasnip::get_model_env()
# }
```
