# Process Predictor Input for Keras (Functional API)

Preprocesses predictor data (`x`) into a format suitable for Keras
models built with the Functional API. Handles both tabular data and
list-columns of arrays (e.g., for images), supporting multiple inputs.

## Usage

``` r
process_x_functional(x)
```

## Arguments

- x:

  A data frame or matrix of predictors.

## Value

A list containing:

- `x_proc`: The processed predictor data (matrix or array, or list of
  arrays).

- `input_shape`: The determined input shape(s) for the Keras model.
