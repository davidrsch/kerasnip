# Predict Confidence Intervals for Classification via LLA

Predict-time entry point for classification confidence intervals. Called
by parsnip via `c(pkg = "kerasnip", fun = "laplace_conf_int_cls")`.

For each output, this uses Monte Carlo sampling from the Laplace
posterior over logits, transforms to probability scale via sigmoid /
softmax, and returns per-class quantile-based intervals.

## Usage

``` r
laplace_conf_int_cls(object, x, laplace_data, lvl, level = 0.95)
```

## Arguments

- object:

  The raw Keras model (from `object$fit$fit`).

- x:

  Processed predictor data (matrix or array).

- laplace_data:

  A named list of Laplace posterior data (from `object$fit$laplace`).

- lvl:

  Character vector of class level names (from `object$fit$lvl`).

- level:

  Confidence level (default 0.95).

## Value

For single-output: a data frame with per-class `.pred_lower_Level` and
`.pred_upper_Level` columns.
