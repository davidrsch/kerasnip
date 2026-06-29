# Predict Confidence Intervals via Last-Layer Laplace Approximation

Predict-time entry point for regression confidence intervals. Called by
parsnip via `c(pkg = "kerasnip", fun = "laplace_conf_int_reg")`.

For each output in the model, this builds the per-sample epistemic
variance (uncertainty on `E[Y|X]`) from the stored Laplace posterior and
returns symmetric Normal-based intervals at the requested confidence
level.

## Usage

``` r
laplace_conf_int_reg(object, x, laplace_data, level = 0.95)
```

## Arguments

- object:

  The raw Keras model (from `object$fit$fit`).

- x:

  Processed predictor data (matrix or array).

- laplace_data:

  A named list of Laplace posterior data, one entry per output (from
  `object$fit$laplace`). Each entry contains `h_diag`, `tau`,
  `sigma_sq_noise`, `n_training`, and `combined_model`.

- level:

  Confidence level (default 0.95). Passed through from
  `predict(..., type = "conf_int", level = 0.95)`.

## Value

A named list of matrices (one per output), each with columns `.pred`,
`.pred_lower`, and `.pred_upper`.
