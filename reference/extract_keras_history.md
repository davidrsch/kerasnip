# Extract Keras Training History

Extracts and returns the training history from a `parsnip` `model_fit`
object created by `kerasnip`.

## Usage

``` r
extract_keras_history(object)
```

## Arguments

- object:

  A `model_fit` object produced by a `kerasnip` specification.

## Value

A `keras_training_history` object. You can call
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) on this object
to visualize the learning curves.

## Details

Extract Keras Training History

The history object contains the metrics recorded during model training,
such as loss and accuracy, for each epoch. This is highly useful for
visualizing the training process and diagnosing issues like overfitting.
The returned object can be plotted directly.

## See also

keras_evaluate, extract_keras_model
