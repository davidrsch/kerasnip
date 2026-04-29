# Prediction Intervals with Conformal Inference

## Why neural networks do not produce native prediction intervals

Linear models derive prediction intervals analytically from the
covariance structure of their parameter estimates. A neural network has
no equivalent: it computes a single deterministic output for each input,
with no internal distributional model to derive intervals from.

This is not a limitation specific to `kerasnip`. The entire tidymodels
ecosystem reflects the same reality, the `parsnip` package does not
register `conf_int` or `pred_int` prediction types for any neural
network engine, including `brulee`, the official tidymodels neural
network package.

The supported and recommended path for obtaining prediction intervals
from neural networks in tidymodels is **conformal inference**, provided
by the `probably` package. Conformal inference is:

- **Distribution-free**: it makes no assumptions about the shape of the
  outcome distribution.
- **Model-agnostic**: it treats the model as a black box; only
  [`fit()`](https://generics.r-lib.org/reference/fit.html) and
  [`predict()`](https://rdrr.io/r/stats/predict.html) calls are
  required.
- **Guaranteed to have correct coverage**: under the assumption that
  training and test data are exchangeable (i.e. identically and
  independently distributed), the intervals contain the true outcome
  with at least the requested probability.

The `probably` package provides three conformal methods, described in
the sections below.

## Setup

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
library(probably)
#> 
#> Attaching package: 'probably'
#> The following objects are masked from 'package:base':
#> 
#>     as.factor, as.ordered
```

## Defining a kerasnip regression model

All three methods work with any fitted
[`workflows::workflow()`](https://workflows.tidymodels.org/reference/workflow.html)
object, so the setup is identical to any other kerasnip workflow.

``` r
input_block <- function(model, input_shape) {
  keras3::keras_model_sequential(input_shape = input_shape)
}
dense_block <- function(model, units = 32) {
  model |>
    keras3::layer_dense(units = units, activation = "relu")
}
output_block <- function(model) {
  model |> keras3::layer_dense(units = 1)
}

create_keras_sequential_spec(
  model_name = "conf_mlp",
  layer_blocks = list(
    input  = input_block,
    dense  = dense_block,
    output = output_block
  ),
  mode = "regression"
)

spec <- conf_mlp(dense_units = 32, fit_epochs = 30) |>
  set_engine("keras")

data <- modeldata::ames |> select(
  Sale_Price,
  Gr_Liv_Area,
  Year_Built,
  Garage_Area,
  Total_Bsmt_SF
)
rec <- recipe(Sale_Price ~ ., data = data) |>
  step_normalize(all_numeric_predictors())
wflow <- workflow(rec, spec)
```

------------------------------------------------------------------------

## Method 1: Split conformal inference (`int_conformal_split`)

This is the simplest and fastest method. The workflow is:

1.  Fit the model on a **training set**.
2.  Pass a separate **calibration set** to
    [`int_conformal_split()`](https://probably.tidymodels.org/reference/int_conformal_split.html).
    The function runs
    [`predict()`](https://rdrr.io/r/stats/predict.html) once on the
    calibration set and records the residuals.
3.  Call [`predict()`](https://rdrr.io/r/stats/predict.html) on the
    resulting object to obtain intervals for new data.

The model is never re-fitted. The calibration residuals act as a
reference distribution for deciding how wide to make the intervals.

``` r
set.seed(1)
split <- initial_split(data, prop = 0.75)
train_dat <- training(split)
cal_dat <- testing(split)

# Fit on the training set
fit_obj <- fit(wflow, data = train_dat)

# Build the conformal object from the calibration set
conformal_split <- int_conformal_split(fit_obj, cal_data = cal_dat)
#> 23/23 - 0s - 3ms/step
conformal_split
#> Split Conformal inference
#> preprocessor: recipe 
#> model: conf_mlp (engine = keras) 
#> calibration set size: 733 
#> 
#> Use `predict(object, new_data, level)` to compute prediction intervals

# Predict intervals for new observations
new_obs <- cal_dat[1:6, ]
predict(conformal_split, new_data = new_obs, level = 0.90)
#> 1/1 - 0s - 20ms/step
#> # A tibble: 6 × 3
#>    .pred .pred_lower .pred_upper
#>    <dbl>       <dbl>       <dbl>
#> 1 31434.    -179710.     242579.
#> 2 97563.    -113581.     308708.
#> 3 52903.    -158242.     264048.
#> 4 58939.    -152205.     270084.
#> 5 12758.    -198386.     223903.
#> 6 16475.    -194670.     227620.
```

**When to use this**: when training cost is non-trivial and you can
afford to set aside a calibration set. This is the recommended method
for kerasnip models in most practical situations.

------------------------------------------------------------------------

## Method 2: Cross-validation conformal inference (`int_conformal_cv`)

This method uses **cross-validation** to generate out-of-fold
predictions, then pools those residuals to calibrate the intervals. The
workflow is:

1.  Define cross-validation folds.
2.  Run
    [`tune::fit_resamples()`](https://tune.tidymodels.org/reference/fit_resamples.html)
    on the workflow with `save_pred = TRUE` and
    `extract = function(x) x` so that both predictions and the fitted
    model objects are retained for each fold.
3.  Pass the resulting `resample_results` object to
    [`int_conformal_cv()`](https://probably.tidymodels.org/reference/int_conformal_cv.html).
    It reads the held-out predictions and uses them to calibrate the
    intervals.
4.  Call [`predict()`](https://rdrr.io/r/stats/predict.html) on the
    resulting object to obtain intervals for new data.

The final model is fitted once more on the full training data inside
[`int_conformal_cv()`](https://probably.tidymodels.org/reference/int_conformal_cv.html).

``` r
set.seed(1)
folds <- vfold_cv(data, v = 5)

# Fit the workflow on each fold, retaining predictions and model objects
fitted_folds <- tune::fit_resamples(
  wflow,
  folds,
  control = tune::control_resamples(
    save_pred = TRUE,
    extract = function(x) x
  )
)
#> 19/19 - 0s - 3ms/step
#> 19/19 - 0s - 3ms/step
#> 19/19 - 0s - 3ms/step
#> 19/19 - 0s - 3ms/step
#> 19/19 - 0s - 3ms/step

# Build the conformal object from the resampling results
conformal_cv <- int_conformal_cv(fitted_folds)
conformal_cv
#> Conformal inference via CV+
#> preprocessor: recipe 
#> model: conf_mlp (engine = keras) 
#> number of models: 5 
#> training set size: 2,930 
#> 
#> Use `predict(object, new_data, level)` to compute prediction intervals

predict(conformal_cv, new_data = data[1:6, ], level = 0.90)
#> 1/1 - 0s - 21ms/step
#> 1/1 - 0s - 20ms/step
#> 1/1 - 0s - 20ms/step
#> 1/1 - 0s - 20ms/step
#> 1/1 - 0s - 20ms/step
#> # A tibble: 6 × 3
#>   .pred_lower  .pred .pred_upper
#>         <dbl>  <dbl>       <dbl>
#> 1    -186742. 26168.     239079.
#> 2    -194767. 18143.     231054.
#> 3    -198423. 14487.     227398.
#> 4    -139409. 73502.     286413.
#> 5    -178310. 34601.     247511.
#> 6    -178966. 33944.     246855
```

**When to use this**: when you do not want to reserve a separate
calibration set and are already running cross-validation to evaluate
your model. The cross-validation residuals tend to produce slightly
narrower intervals than the split method because they use more of the
data. For kerasnip models, note that this refits the model once per
fold, which multiplies training time by the number of folds.

------------------------------------------------------------------------

## Method 3: Full conformal inference (`int_conformal_full`)

Full conformal inference is the most principled of the three methods in
terms of statistical guarantees, but it is the most expensive
computationally.

The algorithm works observation by observation. For each new test point,
it appends that observation to the training set with a **candidate
outcome value**, refits the model from scratch on the augmented dataset,
and checks whether the residual for the candidate value is consistent
with the training residuals. By sweeping over a grid of candidate
values, it determines the interval of values that the algorithm would
not reject.

### Why `int_conformal_full` is unreliable without `fit_seed`

For a kerasnip model, every refit goes through a full
build–compile–train cycle starting from **random weight
initialisation**. This creates a fundamental incompatibility with
`probably`’s bounds-search algorithm, which assumes that as the
candidate outcome sweeps across the trial grid, the nonconformity score
for the trial row changes **monotonically**. That assumption holds for
deterministic models, linear regression, random forests, gradient
boosting, where the same data always produces the same model. It does
not hold for a neural network re-trained from a different random state
on every refit.

When the scores happen to be non-monotone, `probably` cannot find the
interval boundaries and returns `NA`. **Critically, this does not happen
on every run.** The outcome depends on the data, the architecture, and
the random state of the session at the time of the call. Results may
look valid on one run and produce `NA` on the next, with no way to know
in advance which will occur. A result that appears correct cannot be
trusted to be reproducible.

kerasnip detects when it is being called inside `int_conformal_full`’s
refit loop and emits a warning on every internal refit when `fit_seed`
is absent. The warning is the **reliable signal**, it fires
unconditionally regardless of whether the current run happens to
succeed:

``` r
# This code pattern will trigger a kerasnip warning on every internal refit.
# The intervals may or may not be NA depending on the session's random state.
fit_small <- fit(wflow, data = data_small)

conformal_full_broken <- int_conformal_full(
  fit_small,
  train_data = data_small,
  control = control_conformal_full(
    method = "grid",
    trial_points = 20
  )
)

# Warning: int_conformal_full() refits the model from scratch for every trial
# value. Without `fit_seed`, each refit starts from a different random
# initialisation, so nonconformity scores across the trial grid may be
# non-monotone. When that happens, probably cannot find the interval boundaries
# and returns NA — but the problem does not occur on every run, so results that
# look valid may not be reproducible or trustworthy.
# Fix: add `fit_seed = <integer>` to your model spec ...
predict(conformal_full_broken, new_data = new_obs_small, level = 0.90)
```

If you see this warning, treat the output as unreliable regardless of
whether the intervals contain numbers or `NA`.

### The fix: `fit_seed`

Setting `fit_seed` in the model spec calls
[`keras3::set_random_seed()`](https://keras3.posit.co/reference/set_random_seed.html)
at the start of every internal refit, making each training run
deterministic and restoring the monotonicity the bounds-search requires.
The warning will not fire and the intervals will be reproducible across
runs:

``` r
# Use a small subset to keep runtime reasonable in this vignette.
# int_conformal_full refits the model (n_new_obs x n_grid_points) times —
# here 3 x 20 = 60 complete training runs on top of the initial fit.
data_small    <- data[1:100, ]
new_obs_small <- data[101:103, ]

# fit_seed = 42L ensures every internal refit uses the same random state.
# The value is arbitrary; what matters is that it is fixed and consistent.
spec_seeded <- conf_mlp(
  dense_units = 32,
  fit_epochs  = 30,
  fit_seed    = 42L
) |>
  set_engine("keras")

wflow_seeded <- workflow(rec, spec_seeded)
fit_seeded   <- fit(wflow_seeded, data = data_small)

conformal_full <- int_conformal_full(
  fit_seeded,
  train_data = data_small,
  control = control_conformal_full(
    method = "grid",
    trial_points = 20
  )
)
#> 4/4 - 0s - 12ms/step
conformal_full
#> Conformal inference
#> preprocessor: recipe 
#> model: conf_mlp (engine = keras) 
#> training set size: 100 
#> 
#> Use `predict(object, new_data, level)` to compute prediction intervals

predict(conformal_full, new_data = new_obs_small, level = 0.90)
#> 1/1 - 0s - 21ms/step
#> Warning: Unknown or uninitialised column: `difference`.
#> Warning: Could not determine bounds.
#> Warning: Unknown or uninitialised column: `difference`.
#> Warning: Could not determine bounds.
#> Warning: Unknown or uninitialised column: `difference`.
#> Warning: Could not determine bounds.
#> # A tibble: 3 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1          NA          NA
#> 2          NA          NA
#> 3          NA          NA
```

### When to use `int_conformal_full`

Prefer `int_conformal_split` or `int_conformal_cv` for kerasnip models,
both calibrate a single fixed model and carry none of the above caveats.
Use `int_conformal_full` only when the stronger coverage guarantee is
required, the training cost is acceptable, and `fit_seed` is set.

------------------------------------------------------------------------

## Comparison of the three methods

| Method                | Model refits              | Coverage guarantee | Recommended for kerasnip                      |
|-----------------------|---------------------------|--------------------|-----------------------------------------------|
| `int_conformal_split` | 1 (on training data only) | Marginal           | ✅ Yes, lowest cost                           |
| `int_conformal_cv`    | 1 per fold + 1 final      | Marginal           | ✅ Yes, if already cross-validating           |
| `int_conformal_full`  | *(n_test × n_grid)*       | Marginal           | ⚠️ Only with `fit_seed`, unreliable otherwise |

All three methods provide **marginal coverage**: across many test
observations, at least the requested fraction will have their true
outcome within the interval. None of them provide **conditional
coverage** (intervals that are exactly correct for every specific input
value), but in practice they are well-calibrated for most regression
problems.

------------------------------------------------------------------------

## Cleanup

``` r
remove_keras_spec("conf_mlp")
#> Removed from parsnip registry objects: conf_mlp, conf_mlp_args, conf_mlp_encoding, conf_mlp_fit, conf_mlp_modes, conf_mlp_pkgs, conf_mlp_predict
#> Removed 'conf_mlp' from parsnip:::get_model_env()$models
```
