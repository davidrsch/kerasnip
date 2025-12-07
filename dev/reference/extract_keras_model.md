# Extract Keras Model from a Fitted Kerasnip Object

Extracts and returns the underlying Keras model object from a `parsnip`
`model_fit` object created by `kerasnip`.

## Usage

``` r
extract_keras_model(object)
```

## Arguments

- object:

  A `model_fit` object produced by a `kerasnip` specification.

## Value

The raw Keras model object (`keras_model`).

## Details

Extract the Raw Keras Model from a Kerasnip Fit

This is useful when you need to work directly with the Keras model
object for tasks like inspecting layer weights, creating custom plots,
or passing it to other Keras-specific functions.

## See also

keras_evaluate, extract_keras_history
