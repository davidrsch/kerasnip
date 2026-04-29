# Saving and Reloading Fitted Kerasnip Workflows

## Overview

Keras models are backed by Python objects managed by TensorFlow/JAX.
These objects live in the current R session and are represented as
external pointers (`externalptr`) that become invalid as soon as the
session ends, or even within the same session after
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) /
[`readRDS()`](https://rdrr.io/r/base/readRDS.html).

`kerasnip` handles this transparently so that fitted workflows can be
saved, reloaded, and used for prediction without any manual restoration
steps.

## Quick workflow example

Before discussing the details, here is the full persistence workflow:

``` r
library(kerasnip)
library(tidymodels)
#> ── Attaching packages ────────────────────────────────────── tidymodels 1.5.0 ──
#> ✔ broom        1.0.12     ✔ recipes      1.3.2 
#> ✔ dials        1.4.3      ✔ rsample      1.3.2 
#> ✔ dplyr        1.2.1      ✔ tailor       0.1.0 
#> ✔ ggplot2      4.0.3      ✔ tidyr        1.3.2 
#> ✔ infer        1.1.0      ✔ tune         2.1.0 
#> ✔ modeldata    1.5.1      ✔ workflows    1.3.0 
#> ✔ parsnip      1.5.0      ✔ workflowsets 1.1.1 
#> ✔ purrr        1.2.2      ✔ yardstick    1.4.0
#> ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
#> ✖ purrr::discard() masks scales::discard()
#> ✖ dplyr::filter()  masks stats::filter()
#> ✖ dplyr::lag()     masks stats::lag()
#> ✖ recipes::step()  masks stats::step()
library(keras3)
#> 
#> Attaching package: 'keras3'
#> The following object is masked from 'package:yardstick':
#> 
#>     get_weights
#> The following object is masked from 'package:infer':
#> 
#>     generate

# 1. Define Layer Blocks (Required by kerasnip)
# The first block must initialize the sequential model
input_block <- function(model, input_shape) {
  keras_model_sequential(input_shape = input_shape)
}

# Hidden layer block
dense_block <- function(model, units = 32) {
  model |> layer_dense(units = units, activation = "relu")
}

# Output layer block (units = 1 for regression)
output_block <- function(model, num_classes) {
  model |> layer_dense(units = 1)
}

# 2. Generate the parsnip specification
create_keras_sequential_spec(
  model_name = "my_mlp",
  layer_blocks = list(
    input = input_block,
    hidden = dense_block,
    output = output_block
  ),
  mode = "regression"
)

# 3. Use the newly created 'my_mlp' function
mod_spec <- my_mlp(fit_epochs = 10) |> 
  set_engine("keras")

# 4. Standard tidymodels workflow
rec_spec <- recipe(mpg ~ ., data = mtcars) |> 
  step_normalize(all_predictors())

fit_wf <- workflow() |> 
  add_recipe(rec_spec) |> 
  add_model(mod_spec) |> 
  fit(data = mtcars)

# Predict
new_data <- mtcars[1:3, ]
predict(fit_wf, new_data)
#> 1/1 - 0s - 34ms/step
#> # A tibble: 3 × 1
#>   .pred
#>   <dbl>
#> 1  1.82
#> 2  1.66
#> 3  3.41
```

The first call to predict() detects that the Python pointer is invalid
and restores the model from the stored bytes automatically.

## What kerasnip does behind the scenes

`kerasnip` handles persistence automatically:

- At **fit time**, the Keras model is serialized to a raw byte vector
  (`.keras`format) and stored alongside the parsnip `model_fit` object.
- At **predict time**, if the Python pointer is detected as invalid,
  [`predict()`](https://rdrr.io/r/stats/predict.html) automatically
  restores the model from those bytes before dispatching.
- The parsnip model specification is also re-registered if it is missing
  from the session (e.g. after a fresh start).

This means you can use the persistence strategy that best suits your
workflow without any extra boilerplate.

## Strategy 1: Plain `saveRDS()` / `readRDS()`

For most use cases: sharing a model file with a colleague, caching a fit
between R sessions, or checkpointing during development; plain RDS is
the simplest approach.

``` r
library(kerasnip)
library(workflows)
library(parsnip)
library(recipes)

# --- Save ---
saveRDS(fit_wf, "my_model.rds")

# --- Reload in the same or a new R session ---
library(kerasnip)
fit_wf <- readRDS("my_model.rds")

# predict() restores the Keras model from bytes automatically
predictions <- predict(fit_wf, new_data = new_data)
#> 1/1 - 0s - 34ms/step
predictions
#> # A tibble: 3 × 1
#>   .pred
#>   <dbl>
#> 1  1.82
#> 2  1.66
#> 3  3.41
```

There is nothing special to do after
[`readRDS()`](https://rdrr.io/r/base/readRDS.html). The first call to
[`predict()`](https://rdrr.io/r/stats/predict.html) detects the invalid
pointer, restores the model from the stored bytes, and then proceeds
normally.

## Strategy 2: `bundle` / `unbundle`

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
#> 1/1 - 0s - 33ms/step
predictions
#> # A tibble: 3 × 1
#>   .pred
#>   <dbl>
#> 1  1.82
#> 2  1.66
#> 3  3.41
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
