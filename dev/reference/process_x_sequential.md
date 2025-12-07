# Process Predictor Input for Keras

Preprocesses predictor data (`x`) into a format suitable for Keras
models. Handles both tabular data and list-columns of arrays (e.g., for
images).

## Usage

``` r
process_x_sequential(x)
```

## Arguments

- x:

  A data frame or matrix of predictors.

## Value

A list containing:

- `x_proc`: The processed predictor data (matrix or array).

- `input_shape`: The determined input shape for the Keras model.
