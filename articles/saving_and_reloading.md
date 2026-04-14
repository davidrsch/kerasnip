# Saving and Reloading Fitted Kerasnip Workflows

## Overview

Keras models are backed by Python objects managed by TensorFlow/JAX.
These objects live in the current R session and are represented as
external pointers (`externalptr`) that become invalid as soon as the
session ends — or even within the same session after
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) /
[`readRDS()`](https://rdrr.io/r/base/readRDS.html).

`kerasnip` handles this transparently:

- At **fit time**, the Keras model is serialized to a raw byte vector
  (`.keras` format) and stored alongside the parsnip `model_fit` object.
- At **predict time**, if the Python pointer is detected as invalid,
  [`predict()`](https://rdrr.io/r/stats/predict.html) automatically
  restores the model from those bytes before dispatching.
- The parsnip model specification is also re-registered if it is missing
  from the session (e.g. after a fresh start).

This means you can use the persistence strategy that best suits your
workflow without any extra boilerplate.

## Strategy 1 — Plain `saveRDS()` / `readRDS()`

For most use cases — sharing a model file with a colleague, caching a
fit between R sessions, or checkpointing during development — plain RDS
is the simplest approach.

``` r
library(kerasnip)
library(workflows)
library(parsnip)
library(recipes)

# Assume `fit_wf` is a fitted workflow produced by kerasnip
# (see vignette("workflows_sequential") or vignette("workflows_functional"))

# --- Save ---
saveRDS(fit_wf, "my_model.rds")

# --- Reload in the same or a new R session ---
library(kerasnip)
fit_wf <- readRDS("my_model.rds")

# predict() restores the Keras model from bytes automatically
predictions <- predict(fit_wf, new_data = new_data)
```

There is nothing special to do after
[`readRDS()`](https://rdrr.io/r/base/readRDS.html). The first call to
[`predict()`](https://rdrr.io/r/stats/predict.html) detects the invalid
pointer, restores the model from the stored bytes, and then proceeds
normally.

## Strategy 2 — `bundle` / `unbundle`

The [`bundle`](https://rstudio.github.io/bundle/) package provides a
standardized serialization interface used by `vetiver`, `plumber`, and
other MLOps tools. It is the right choice when:

- You are deploying a model to a `vetiver` API or a Docker container.
- You want a self-contained, version-controlled artifact that does not
  rely on any R session state.
- You are sharing a model across machines with different R library
  paths.

``` r
library(kerasnip)
library(bundle)
library(workflows)

# --- Save ---
bundled <- bundle(fit_wf)
saveRDS(bundled, "my_model_bundle.rds")

# --- Reload in any R session ---
library(kerasnip)
library(bundle)
bundled <- readRDS("my_model_bundle.rds")
fit_wf <- unbundle(bundled)

predictions <- predict(fit_wf, new_data = new_data)
```

## Comparison

|                               | `saveRDS` / `readRDS` | `bundle` / `unbundle` |
|-------------------------------|-----------------------|-----------------------|
| Works across sessions         | ✅                    | ✅                    |
| Works across machines         | ✅ (same R library)   | ✅                    |
| `vetiver` / Docker compatible | ❌                    | ✅                    |
| Extra dependency needed       | ❌                    | `bundle` package      |
| Code complexity               | Minimal               | Minimal               |

## What happens under the hood

When `kerasnip` fits a model, the generic fit function calls
`keras_model_to_bytes()`, which writes the model to a temporary `.keras`
file using
[`keras3::save_model()`](https://keras3.posit.co/reference/save_model.html)
and reads the bytes back into R:

``` r
# Simplified version of what happens inside generic_sequential_fit()
keras_bytes <- keras_model_to_bytes(model)
# keras_bytes is a raw vector stored in object$fit$keras_bytes
```

When [`predict()`](https://rdrr.io/r/stats/predict.html) is called on a
reloaded object,
[`predict.kerasnip_model_fit()`](https://davidrsch.github.io/kerasnip/reference/predict.kerasnip_model_fit.md)
runs:

``` r
# Simplified version of predict.kerasnip_model_fit()
if (!is.null(object$fit$keras_bytes)) {
    is_valid <- tryCatch(
        {
            reticulate::py_validate_xptr(object$fit$fit)
            TRUE
        },
        error = function(e) FALSE
    )
    if (!is_valid) {
        object$fit$fit <- keras_model_from_bytes(object$fit$keras_bytes)
    }
}
```

If `keras_model_to_bytes()` fails (e.g. if the model was compiled with a
non-serialisable custom object), a warning is issued at fit time and
`keras_bytes` is set to `NULL`. In that case,
[`predict()`](https://rdrr.io/r/stats/predict.html) after reload will
fail with a clear error.
