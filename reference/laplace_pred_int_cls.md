# Predict Prediction Intervals for Classification via LLA

Predict-time entry point for classification prediction intervals. Called
by parsnip via `c(pkg = "kerasnip", fun = "laplace_pred_int_cls")`.

After computing epistemic probability samples (as in conf_int), this
draws class labels from `Bernoulli(p_sample)` or `Categorical(p_sample)`
and returns per-class quantiles of the resulting indicator samples —
matching the posterior predictive behaviour of Stan and BART
classification engines.

## Usage

``` r
laplace_pred_int_cls(object, x, laplace_data, lvl, level = 0.95)
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
