# Post-Processing Multi-Output and Multistep Models with tailor and probably

## Why single-output models “just work” and multi-output/multistep models don’t

`tailor` and `probably` are the tidymodels tools for post-processing
model predictions: calibrating probabilities, adjusting classification
thresholds, and building conformal prediction intervals. Both packages
are built around a single assumption: **one outcome column, one estimate
column**.
[`tailor::fit()`](https://generics.r-lib.org/reference/fit.html) selects
them with `[[`, which requires exactly one column each.

For a kerasnip model with a single outcome, this is exactly what
[`predict()`](https://rdrr.io/r/stats/predict.html) already produces
(`.pred` for regression; `.pred_class`/`.pred_<level>` for
classification), so `tailor` and `probably` work unmodified — see the
[Prediction Intervals with Conformal
Inference](https://davidrsch.github.io/kerasnip/articles/conformal_intervals.md)
vignette for `probably`, and the examples below for `tailor`.

kerasnip also supports two shapes that go beyond a single outcome, and
neither fits `tailor`/`probably`’s assumption:

- **Multi-output models** (a recipe like `output_1 + output_2 ~ .`, each
  outcome its own Keras head) produce `.pred_output_1`,
  `.pred_output_2`, … — this is the standard
  `parsnip::maybe_multivariate()` shape for multivariate regression, not
  something kerasnip invented, but downstream post-processing tooling
  hasn’t caught up to it for *any* engine yet.
- **Multistep forecasting models** (see
  [`vignette("multistep_forecasting")`](https://davidrsch.github.io/kerasnip/articles/multistep_forecasting.md))
  produce a nested `.pred` list-column: one inner tibble per row,
  holding `.step` and the forecasted value at each step.
  `tailor::check_variable_type()` requires
  [`is.numeric()`](https://rdrr.io/r/base/numeric.html) on the
  outcome/estimate columns, which a list-column fails outright.

kerasnip cannot make
[`workflows::add_tailor()`](https://workflows.tidymodels.org/reference/add_tailor.html)
work directly on either shape —
[`tailor::fit()`](https://generics.r-lib.org/reference/fit.html)’s
single-column selection is baked into its own code, not something
kerasnip’s prediction format can work around. Instead, kerasnip
provides:

- [`kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_output_view.md)
  /
  [`kerasnip_step_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_step_view.md):
  present one output (or one forecast step) as an ordinary single-output
  fit, so you can use `tailor`/`probably` exactly as documented, one
  output/step at a time.
- [`kerasnip_add_tailor()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_add_tailor.md):
  a
  [`workflows::add_tailor()`](https://workflows.tidymodels.org/reference/add_tailor.html)-alike
  built on top of the views, for when you want the tailor trained and
  applied automatically as part of
  [`fit()`](https://generics.r-lib.org/reference/fit.html)/[`predict()`](https://rdrr.io/r/stats/predict.html).

## Setup

``` r

library(kerasnip)
library(tidymodels)
#> ── Attaching packages ────────────────────────────────────── tidymodels 1.5.0 ──
#> ✔ broom        1.0.13     ✔ recipes      1.3.3 
#> ✔ dials        1.4.4      ✔ rsample      1.3.2 
#> ✔ dplyr        1.2.1      ✔ tailor       0.1.0 
#> ✔ ggplot2      4.0.3      ✔ tidyr        1.3.2 
#> ✔ infer        1.1.0      ✔ tune         2.1.0 
#> ✔ modeldata    1.5.1      ✔ workflows    1.3.0 
#> ✔ parsnip      1.6.0      ✔ workflowsets 1.1.1 
#> ✔ purrr        1.2.2      ✔ yardstick    1.4.0
#> ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
#> ✖ purrr::discard() masks scales::discard()
#> ✖ dplyr::filter()  masks stats::filter()
#> ✖ dplyr::lag()     masks stats::lag()
#> ✖ recipes::step()  masks stats::step()
library(tailor)
library(probably)
#> 
#> Attaching package: 'probably'
#> The following objects are masked from 'package:base':
#> 
#>     as.factor, as.ordered
```

------------------------------------------------------------------------

## Multi-output models

### Defining a multi-output regression workflow

``` r

input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
dense_block <- function(tensor, units = 16) {
  tensor |> keras3::layer_dense(units = units, activation = "relu")
}
output_1_block <- function(tensor) keras3::layer_dense(tensor, units = 1, name = "temperature")
output_2_block <- function(tensor) keras3::layer_dense(tensor, units = 1, name = "humidity")

create_keras_functional_spec(
  model_name = "climate_mlp",
  layer_blocks = list(
    main_input = input_block,
    dense      = inp_spec(dense_block, "main_input"),
    temperature = inp_spec(output_1_block, "dense"),
    humidity    = inp_spec(output_2_block, "dense")
  ),
  mode = "regression"
)

spec <- climate_mlp(dense_units = 16, fit_epochs = 30) |>
  set_engine("keras")

set.seed(1)
n <- 300
climate_data <- tibble(
  pressure    = rnorm(n),
  wind_speed  = rnorm(n),
  temperature = pressure + rnorm(n, sd = 0.2),
  humidity    = -0.5 * wind_speed + rnorm(n, sd = 0.2)
)

rec <- recipe(temperature + humidity ~ pressure + wind_speed, data = climate_data)

split <- initial_split(climate_data, prop = 0.7)
train_dat <- training(split)
cal_dat   <- testing(split)

wflow <- workflow(rec, spec)
fit_obj <- fit(wflow, data = train_dat)
#> 7/7 - 0s - 7ms/step
#> 7/7 - 0s - 7ms/step

predict(fit_obj, new_data = cal_dat[1:5, ])
#> 1/1 - 0s - 37ms/step
#> # A tibble: 5 × 2
#>   .pred_temperature .pred_humidity
#>               <dbl>          <dbl>
#> 1             0.356         -0.880
#> 2            -0.869         -0.842
#> 3             0.553          0.499
#> 4             0.480         -0.106
#> 5             0.781          0.623
```

Both outcomes come back from a single
[`predict()`](https://rdrr.io/r/stats/predict.html) call,
`.pred_temperature`/`.pred_humidity` — exactly parsnip’s own convention
for multivariate regression, and exactly what breaks
`tailor`/`probably`, which need one estimate column to work with.

### `kerasnip_output_view()`: one output as a standard single-output fit

``` r

temp_view <- kerasnip_output_view(fit_obj, "temperature")
predict(temp_view, new_data = cal_dat[1:5, ])
#> 1/1 - 0s - 21ms/step
#> # A tibble: 5 × 1
#>    .pred
#>    <dbl>
#> 1  0.356
#> 2 -0.869
#> 3  0.553
#> 4  0.480
#> 5  0.781
```

`temp_view` behaves like an ordinary single-output fit to anything that
calls [`predict()`](https://rdrr.io/r/stats/predict.html) on it. That is
enough for manual `tailor` usage:

``` r

cal_preds <- predict(temp_view, new_data = cal_dat)
#> 3/3 - 0s - 13ms/step
cal_data_for_tailor <- bind_cols(temperature = cal_dat$temperature, cal_preds)

tlr <- tailor() |> adjust_numeric_calibration(method = "linear")
tlr_fit <- fit(tlr, cal_data_for_tailor, outcome = temperature, estimate = .pred)
#> Registered S3 method overwritten by 'butcher':
#>   method                 from    
#>   as.character.dev_topic generics

new_preds <- predict(temp_view, new_data = cal_dat[1:5, ])
#> 1/1 - 0s - 21ms/step
predict(tlr_fit, new_preds)
#> # A tibble: 5 × 1
#>    .pred
#>    <dbl>
#> 1  0.340
#> 2 -0.828
#> 3  0.515
#> 4  0.450
#> 5  0.722
```

It is also enough for `probably`’s conformal methods, because
[`kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_output_view.md)
implements
[`hardhat::extract_mold()`](https://hardhat.tidymodels.org/reference/hardhat-extract.html)
and
[`generics::augment()`](https://generics.r-lib.org/reference/augment.html)
for the view, which is all
[`probably::int_conformal_split()`](https://probably.tidymodels.org/reference/int_conformal_split.html)
needs:

``` r

conformal <- int_conformal_split(temp_view, cal_data = cal_dat)
#> 3/3 - 0s - 7ms/step
predict(conformal, new_data = cal_dat[1:5, ], level = 0.90)
#> 1/1 - 0s - 21ms/step
#> # A tibble: 5 × 3
#>    .pred .pred_lower .pred_upper
#>    <dbl>       <dbl>       <dbl>
#> 1  0.356     -0.0456       0.757
#> 2 -0.869     -1.27        -0.467
#> 3  0.553      0.152        0.954
#> 4  0.480      0.0790       0.882
#> 5  0.781      0.379        1.18
```

### `probably::int_conformal_full()`: supported, with one documented assumption

[`int_conformal_full()`](https://probably.tidymodels.org/reference/int_conformal_full.html)
refits the model once per candidate value of every new observation. For
a multi-output model, refitting means retraining the *whole* multi-head
network — so what should the *other* output(s) be during that refit, for
a row where they were never observed in the first place?
[`kerasnip_output_view()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_output_view.md)’s
[`int_conformal_full()`](https://probably.tidymodels.org/reference/int_conformal_full.html)
support substitutes the model’s own current point-prediction for the
other output(s) as a placeholder. Because the placeholder equals what
that head already predicts, its loss contribution for that one synthetic
row is ~zero, so it should not be measurably disturbed while the target
head still responds to the candidate value under test. This is a
reasonable choice, not a proven one — treat the resulting intervals
accordingly.

``` r

# Small subset to keep runtime reasonable in this vignette.
small_train <- train_dat[1:40, ]
small_new   <- cal_dat[1:3, ]

fit_small <- fit(wflow, data = small_train)
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
temp_view_small <- kerasnip_output_view(fit_small, "temperature")

conformal_full <- int_conformal_full(
  temp_view_small,
  train_data = small_train,
  control = control_conformal_full(method = "grid", trial_points = 15)
)
#> 2/2 - 0s - 26ms/step
predict(conformal_full, new_data = small_new, level = 0.90)
#> 1/1 - 0s - 21ms/step
#> 1/1 - 0s - 20ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 32ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 28ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 28ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 28ms/step
#> 1/1 - 0s - 21ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 25ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 29ms/step
#> 2/2 - 0s - 25ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 29ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 28ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 25ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> 1/1 - 0s - 21ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 28ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 28ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 28ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 26ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 25ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 28ms/step
#> 2/2 - 0s - 25ms/step
#> 2/2 - 0s - 25ms/step
#> 2/2 - 0s - 29ms/step
#> 2/2 - 0s - 25ms/step
#> 2/2 - 0s - 25ms/step
#> 2/2 - 0s - 28ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 24ms/step
#> 2/2 - 0s - 27ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 23ms/step
#> 2/2 - 0s - 26ms/step
#> # A tibble: 3 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1      -0.544       1.11 
#> 2      -1.41        0.241
#> 3       0.180       1.21
```

Only `method = "grid"` is supported; `"iterative"` relies on
`probably`’s private root-finding internals and is out of scope.

### `kerasnip_add_tailor()`: attach and forget

For routine use,
[`kerasnip_add_tailor()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_add_tailor.md)
wraps the view + fit + splice-back steps into the same
[`add_tailor()`](https://workflows.tidymodels.org/reference/add_tailor.html)-style
workflow you would use for a single-output model — except it targets one
named output, and every other output’s columns pass through untouched:

``` r

tlr2 <- tailor() |> adjust_numeric_calibration(method = "linear")
tailored_wf <- kerasnip_add_tailor(wflow, tlr2, output = "temperature")

tailored_fit <- fit(tailored_wf, data = train_dat, data_calibration = cal_dat)
#> 7/7 - 0s - 8ms/step
#> 7/7 - 0s - 7ms/step
#> 3/3 - 0s - 18ms/step
predict(tailored_fit, new_data = cal_dat[1:5, ])
#> 1/1 - 0s - 22ms/step
#> 1/1 - 0s - 22ms/step
#> # A tibble: 5 × 2
#>   .pred_temperature .pred_humidity
#>               <dbl>          <dbl>
#> 1             0.267        -0.877 
#> 2            -0.760        -0.772 
#> 3             0.478         0.435 
#> 4             0.454        -0.0522
#> 5             0.686         0.556
```

`.pred_temperature` is calibrated; `.pred_humidity` is exactly what a
plain, un-tailored [`predict()`](https://rdrr.io/r/stats/predict.html)
would have returned.

------------------------------------------------------------------------

## Multistep forecasting models

A multistep model (see
[`vignette("multistep_forecasting")`](https://davidrsch.github.io/kerasnip/articles/multistep_forecasting.md))
has a *single* outcome conceptually — but
[`predict()`](https://rdrr.io/r/stats/predict.html) returns it as a
nested `.pred` list-column (one row per sample, one inner tibble per row
holding `.step` and the forecasted value), which `tailor`’s
[`is.numeric()`](https://rdrr.io/r/base/numeric.html) check rejects just
as firmly as a genuine multi-output shape, for an unrelated reason.

``` r

set.seed(42)
n_steps <- 200
timesteps <- 12
horizon <- 4
series <- tibble(value = sin(seq_len(n_steps) / 10) + rnorm(n_steps, sd = 0.05))

rec_step <- recipe(series) |>
  step_lead(value, lead = seq_len(horizon), prefix = "lead_") |>
  step_naomit(starts_with("lead_")) |>
  step_sequence(value, timesteps = timesteps, new_col = "window")

window_input <- function(input_shape) keras3::layer_input(shape = input_shape, name = "window_input")
lstm_block   <- function(tensor, units = 16) tensor |> keras3::layer_lstm(units = units)
step_output  <- function(tensor, units = 1) tensor |> keras3::layer_dense(units = units)

create_keras_functional_spec(
  model_name = "forecast_lstm",
  layer_blocks = list(
    window = window_input,
    lstm   = inp_spec(lstm_block, "window"),
    output = inp_spec(step_output, "lstm")
  ),
  mode = "regression"
)

step_spec <- forecast_lstm(lstm_units = 16, output_units = horizon, fit_epochs = 30) |>
  set_engine("keras")

split_step <- initial_time_split(series, prop = 0.8)
train_series <- training(split_step)
test_series  <- testing(split_step)

step_wflow <- workflow(rec_step, step_spec)
step_fit <- fit(step_wflow, data = train_series)
#> 5/5 - 0s - 36ms/step

# step_sequence() needs `timesteps` rows of leading history to produce a
# single prediction, so a preview slice must include at least that much
# context; this gives 6 rows with a full window.
preview_data <- test_series[seq_len(timesteps + 5), , drop = FALSE]

predict(step_fit, new_data = preview_data)
#> 1/1 - 0s - 95ms/step
#> # A tibble: 6 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [4 × 2]>
#> 2 <tibble [4 × 2]>
#> 3 <tibble [4 × 2]>
#> 4 <tibble [4 × 2]>
#> 5 <tibble [4 × 2]>
#> 6 <tibble [4 × 2]>
```

### `kerasnip_step_view()`: one forecast step as a standard single-output fit

``` r

step_2_view <- kerasnip_step_view(step_fit, step = 2)
predict(step_2_view, new_data = preview_data)
#> 1/1 - 0s - 22ms/step
#> # A tibble: 6 × 1
#>    .pred
#>    <dbl>
#> 1 -0.957
#> 2 -0.978
#> 3 -0.941
#> 4 -0.893
#> 5 -0.855
#> 6 -0.835
```

Unlike a multi-output model, a multistep model’s per-step outcome
columns (`lead_2_value`, …) are recipe-*engineered* from a single raw
column by
[`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
— they do not exist in raw data the way `output_1`/`output_2` do for a
genuine multi-output model.
[`kerasnip_step_truth()`](https://davidrsch.github.io/kerasnip/reference/kerasnip_step_truth.md)
recovers the true future value at a given step by re-baking the fitted
recipe:

``` r

truth <- kerasnip_step_truth(step_2_view, test_series)
head(truth)
#> [1] -0.9081864 -0.9323871 -0.9563833 -0.9850328 -0.8350517 -0.7889974
```

That is enough to calibrate manually, same as the multi-output case:

``` r

preds_step <- predict(step_2_view, new_data = train_series)
#> 5/5 - 0s - 20ms/step
truth_step <- kerasnip_step_truth(step_2_view, train_series)

cal_tbl <- tibble(truth = truth_step, .pred = preds_step$.pred) |>
  filter(!is.na(truth))

tlr_step <- tailor() |> adjust_numeric_calibration(method = "linear")
tlr_step_fit <- fit(tlr_step, cal_tbl, outcome = truth, estimate = .pred)

new_preds_step <- predict(step_2_view, new_data = preview_data)
#> 1/1 - 0s - 22ms/step
predict(tlr_step_fit, new_preds_step)
#> # A tibble: 6 × 1
#>    .pred
#>    <dbl>
#> 1 -0.963
#> 2 -0.984
#> 3 -0.947
#> 4 -0.899
#> 5 -0.861
#> 6 -0.841
```

…and enough for
[`probably::int_conformal_split()`](https://probably.tidymodels.org/reference/int_conformal_split.html),
exactly as with a multi-output view:

``` r

conformal_step <- int_conformal_split(step_2_view, cal_data = train_series)
#> 5/5 - 0s - 5ms/step
predict(conformal_step, new_data = preview_data, level = 0.90)
#> 1/1 - 0s - 22ms/step
#> # A tibble: 6 × 3
#>    .pred .pred_lower .pred_upper
#>    <dbl>       <dbl>       <dbl>
#> 1 -0.957      -1.05       -0.860
#> 2 -0.978      -1.08       -0.880
#> 3 -0.941      -1.04       -0.843
#> 4 -0.893      -0.991      -0.796
#> 5 -0.855      -0.953      -0.758
#> 6 -0.835      -0.932      -0.737
```

[`probably::int_conformal_full()`](https://probably.tidymodels.org/reference/int_conformal_full.html)
is also supported for step views, with a different design from the
multi-output case: a multistep model’s step targets are not independent
raw columns — every `lead_k_value` column is derived from the *same*
single raw column by
[`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md).
Testing a candidate value for one step means writing that candidate into
the raw column at the appropriate future offset, which also (partially)
supplies the targets for the *other* steps forecast from the same
origin; those other steps’ placeholders are the current model’s own
forecast, the same idea as the multi-output case’s “other output(s)”
placeholder. This is only supported when
[`step_lead()`](https://davidrsch.github.io/kerasnip/reference/step_lead.md)
and
[`step_sequence()`](https://davidrsch.github.io/kerasnip/reference/step_sequence.md)
share a single source column, true of the model built above (and every
multistep example in this package).

``` r

# Small subset to keep runtime reasonable in this vignette, but wide enough
# for the residual-variance model to see a representative range of
# predictions — too narrow a range makes it extrapolate wildly for new
# observations outside it. small_new needs at least `timesteps` rows of
# leading context, same as preview_data above.
small_train <- train_series[1:80, , drop = FALSE]
small_new <- test_series[seq_len(timesteps + 2), , drop = FALSE]

fit_small <- fit(step_wflow, data = small_train)
#> 3/3 - 0s - 60ms/step
step_2_view_small <- kerasnip_step_view(fit_small, step = 2)

conformal_step_full <- int_conformal_full(
  step_2_view_small,
  train_data = small_train,
  control = control_conformal_full(method = "grid", trial_points = 10)
)
#> 3/3 - 0s - 58ms/step
predict(conformal_step_full, new_data = small_new, level = 0.90)
#> 1/1 - 0s - 23ms/step
#> 1/1 - 0s - 21ms/step
#> 3/3 - 0s - 58ms/step
#> 3/3 - 0s - 57ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 58ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 58ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 58ms/step
#> 3/3 - 0s - 63ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 61ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 58ms/step
#> 3/3 - 0s - 58ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 61ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 58ms/step
#> 3/3 - 1s - 401ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 60ms/step
#> 3/3 - 0s - 58ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 63ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 61ms/step
#> 3/3 - 0s - 59ms/step
#> 3/3 - 0s - 62ms/step
#> 3/3 - 0s - 61ms/step
#> 3/3 - 0s - 61ms/step
#> 3/3 - 0s - 59ms/step
#> # A tibble: 3 × 2
#>   .pred_lower .pred_upper
#>         <dbl>       <dbl>
#> 1       -1.26      -0.659
#> 2       -1.29      -0.698
#> 3       -1.25      -0.646
```

As with the multi-output case, only `method = "grid"` is supported.

### `kerasnip_add_tailor()` for one forecast step

``` r

tlr_step2 <- tailor() |> adjust_numeric_calibration(method = "linear")
tailored_step_wf <- kerasnip_add_tailor(step_wflow, tlr_step2, step = 2)

tailored_step_fit <- fit(tailored_step_wf, data = train_series)
#> 5/5 - 0s - 39ms/step
#> 5/5 - 0s - 36ms/step
predict(tailored_step_fit, new_data = preview_data)
#> 1/1 - 0s - 24ms/step
#> 1/1 - 0s - 23ms/step
#> # A tibble: 6 × 1
#>   .pred           
#>   <list>          
#> 1 <tibble [4 × 2]>
#> 2 <tibble [4 × 2]>
#> 3 <tibble [4 × 2]>
#> 4 <tibble [4 × 2]>
#> 5 <tibble [4 × 2]>
#> 6 <tibble [4 × 2]>
```

Step 2’s forecasted value is calibrated in every row’s nested tibble;
every other step is left exactly as a plain
[`predict()`](https://rdrr.io/r/stats/predict.html) would have returned
it.

------------------------------------------------------------------------

## Cleanup

``` r

remove_keras_spec("climate_mlp")
#> Removed from parsnip registry objects: climate_mlp, climate_mlp_args, climate_mlp_encoding, climate_mlp_fit, climate_mlp_modes, climate_mlp_pkgs, climate_mlp_predict
#> Removed 'climate_mlp' from parsnip:::get_model_env()$models
remove_keras_spec("forecast_lstm")
#> Removed from parsnip registry objects: forecast_lstm, forecast_lstm_args, forecast_lstm_encoding, forecast_lstm_fit, forecast_lstm_modes, forecast_lstm_pkgs, forecast_lstm_predict
#> Removed 'forecast_lstm' from parsnip:::get_model_env()$models
```
