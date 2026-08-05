# Extract Mold Method for `kerasnip_output_view` Objects

Returns the wrapped workflow's mold with `$outcomes` (and, if present,
`$blueprint$ptypes$outcomes`) sliced down to `x$output`'s single column,
so downstream code that reads `names(extract_mold(x)$outcomes)` (such as
[`probably::int_conformal_split()`](https://probably.tidymodels.org/reference/int_conformal_split.html))
sees a single-outcome fit.

## Usage

``` r
# S3 method for class 'kerasnip_output_view'
extract_mold(x, ...)
```

## Arguments

- x:

  A `kerasnip_output_view`.

- ...:

  Not used.

## Value

A `hardhat` mold with a single outcome column.
