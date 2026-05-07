# Check if a Kerasnip Model Specification Exists

This is an internal helper function to check if a model specification
has been registered in the `parsnip` model environment.

## Usage

``` r
model_exists(model_name)
```

## Arguments

- model_name:

  A character string giving the name of the model specification function
  to check (e.g., "my_mlp").

## Value

A logical value, `TRUE` if the model exists, `FALSE` otherwise.

## Examples

``` r
# \donttest{
if (requireNamespace("parsnip", quietly = TRUE)) {
  library(parsnip)

  # Check for a model that exists in parsnip
  model_exists("mlp")

  # Check for a model that does not exist
  model_exists("non_existent_model")
}
#> [1] FALSE
# }
```
