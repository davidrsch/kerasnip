# =============================================================================
# kerasnip_step_view() tests
#
# A multistep kerasnip model's `.pred` is a nested list-column (one inner
# tibble per row: `.step` + forecasted value), which `tailor`/`probably`
# reject outright (`check_variable_type()` requires `is.numeric()`).
# `kerasnip_step_view()` flattens one forecast step down to a standard
# `.pred` column, and `kerasnip_step_truth()` recovers the true future value
# for that step by re-baking the fitted recipe (the per-step outcome columns
# are recipe-engineered, not present in raw data the way multi-output
# columns are).
# =============================================================================

make_step_view_fit <- function(
  model_name,
  n = 120,
  timesteps = 8,
  horizon = 3,
  fit_epochs = 5
) {
  set.seed(42)
  dat <- tibble::tibble(value = sin(seq_len(n) / 10) + rnorm(n, sd = 0.05))

  rec <- recipe(dat) |>
    step_lead(value, lead = seq_len(horizon), prefix = "lead_") |>
    step_naomit(starts_with("lead_")) |>
    step_sequence(value, timesteps = timesteps, new_col = "window")

  input_block <- function(input_shape) {
    keras3::layer_input(shape = input_shape, name = "window_input")
  }
  lstm_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_lstm(units = units)
  }
  output_block <- function(tensor, units = 1) {
    tensor |> keras3::layer_dense(units = units)
  }

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      window = input_block,
      lstm = inp_spec(lstm_block, "window"),
      output = inp_spec(output_block, "lstm")
    ),
    mode = "regression"
  )

  spec <- get(model_name)(
    output_units = horizon,
    fit_epochs = fit_epochs,
    fit_verbose = 0
  ) |>
    set_engine("keras")

  wf <- workflow(rec, spec)
  fit_obj <- fit(wf, data = dat)

  list(fit_obj = fit_obj, dat = dat, timesteps = timesteps, horizon = horizon)
}

test_that("kerasnip_step_view: errors when x is not a fitted workflow", {
  skip_if_no_keras()

  expect_error(kerasnip_step_view(list(), step = 1), "workflow")
  expect_error(kerasnip_step_view(workflows::workflow(), step = 1), "workflow")
})

test_that("kerasnip_step_view: errors when the model is not multistep", {
  skip_if_no_keras()

  model_name <- "step_view_not_multistep"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  dense_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(tensor) keras3::layer_dense(tensor, units = 1)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      main_input = input_block,
      dense = inp_spec(dense_block, "main_input"),
      output = inp_spec(output_block, "dense")
    ),
    mode = "regression"
  )

  spec <- step_view_not_multistep(fit_epochs = 1) |> set_engine("keras")
  data <- mtcars
  fit_obj <- fit(workflow(recipe(mpg ~ ., data = data), spec), data = data)

  expect_error(kerasnip_step_view(fit_obj, step = 1), "multistep")
})

test_that("kerasnip_step_view: errors on an out-of-range step", {
  skip_if_no_keras()

  model_name <- "step_view_bad_step"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  built <- make_step_view_fit(model_name)

  expect_error(
    kerasnip_step_view(built$fit_obj, step = built$horizon + 10),
    "step"
  )
})

test_that("kerasnip_step_view: multi-variable forecasts require `var`, and validate it", {
  skip_if_no_keras()

  model_name <- "step_view_multi_var"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  set.seed(7)
  n <- 40
  timesteps <- 5
  horizon <- 2
  dat <- tibble::tibble(
    value1 = sin(seq_len(n) / 10) + rnorm(n, sd = 0.05),
    value2 = cos(seq_len(n) / 10) + rnorm(n, sd = 0.05)
  )

  rec <- recipe(dat) |>
    step_lead(value1, value2, lead = seq_len(horizon), prefix = "lead_") |>
    step_naomit(starts_with("lead_")) |>
    step_sequence(value1, value2, timesteps = timesteps, new_col = "window")

  input_block <- function(input_shape) {
    keras3::layer_input(shape = input_shape, name = "window_input")
  }
  lstm_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_lstm(units = units)
  }
  output_block <- function(tensor, units = 1) {
    tensor |> keras3::layer_dense(units = units)
  }

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      window = input_block,
      lstm = inp_spec(lstm_block, "window"),
      output = inp_spec(output_block, "lstm")
    ),
    mode = "regression"
  )

  spec <- get(model_name)(
    output_units = horizon * 2,
    fit_epochs = 1,
    fit_verbose = 0
  ) |>
    set_engine("keras")
  fit_obj <- fit(workflow(rec, spec), data = dat)

  expect_error(
    kerasnip_step_view(fit_obj, step = 1),
    "Multiple forecasted variables found"
  )
  expect_error(
    kerasnip_step_view(fit_obj, step = 1, var = "nonexistent"),
    "must be one of"
  )
})

test_that("kerasnip_step_view: predict() matches the raw nested .pred column", {
  skip_if_no_keras()

  model_name <- "step_view_predict"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  built <- make_step_view_fit(model_name)
  new_data <- built$dat[seq_len(built$timesteps + 5), , drop = FALSE]

  raw_preds <- predict(built$fit_obj, new_data = new_data)
  view_2 <- kerasnip_step_view(built$fit_obj, step = 2)
  view_preds <- predict(view_2, new_data = new_data)

  expect_s3_class(view_preds, "tbl_df")
  expect_equal(names(view_preds), ".pred")
  expect_equal(nrow(view_preds), nrow(raw_preds))

  expected <- vapply(
    raw_preds$.pred,
    function(tbl) tbl$.pred[tbl$.step == 2],
    numeric(1)
  )
  expect_equal(view_preds$.pred, expected)

  # A view built (bypassing kerasnip_step_view()'s own range check) for a
  # step beyond the real fit's forecast horizon: the real nested `.pred`
  # tibbles genuinely have no matching `.step` row.
  bad_view <- structure(
    list(workflow = built$fit_obj, step = built$horizon + 1, var = NULL),
    class = "kerasnip_step_view"
  )
  expect_error(
    predict(bad_view, new_data = new_data),
    "not present in the forecast horizon"
  )
})

test_that("kerasnip_step_truth: recovers the correct future value", {
  skip_if_no_keras()

  model_name <- "step_view_truth"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  built <- make_step_view_fit(model_name)
  n_rows <- built$timesteps + 5
  new_data <- built$dat[seq_len(n_rows), , drop = FALSE]

  view_2 <- kerasnip_step_view(built$fit_obj, step = 2)
  truth <- kerasnip_step_truth(view_2, new_data)

  # step_sequence() drops the leading (timesteps - 1) rows (insufficient
  # window history); step_lead()'s forward shift then gives the truth for
  # each surviving row as the raw value 2 rows ahead, NA past the end of
  # `new_data` (bake() only sees `new_data`'s own extent, not the original
  # series beyond it).
  surviving_idx <- built$timesteps:n_rows
  target_idx <- surviving_idx + 2
  expected <- ifelse(target_idx <= n_rows, new_data$value[target_idx], NA_real_)

  expect_equal(truth, expected)
})

test_that("kerasnip_step_view: manual tailor::adjust_numeric_calibration works per step", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "step_view_cal"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  built <- make_step_view_fit(model_name)
  cal_data <- built$dat[seq_len(built$timesteps + 20), , drop = FALSE]

  view_1 <- kerasnip_step_view(built$fit_obj, step = 1)
  preds <- predict(view_1, new_data = cal_data)
  truth <- kerasnip_step_truth(view_1, cal_data)

  cal_tbl <- tibble::tibble(truth = truth, .pred = preds$.pred)
  cal_tbl <- cal_tbl[stats::complete.cases(cal_tbl), ]

  tlr <- tailor::tailor() |>
    tailor::adjust_numeric_calibration(method = "linear")
  tlr_fit <- fit(tlr, cal_tbl, outcome = truth, estimate = .pred)

  result <- predict(tlr_fit, cal_tbl)

  expect_s3_class(result, "tbl_df")
  expect_true(".pred" %in% names(result))
  expect_true(is.numeric(result$.pred))
  expect_equal(nrow(result), nrow(cal_tbl))
})

test_that("kerasnip_step_view: probably::int_conformal_split works per step", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "step_view_conformal"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  built <- make_step_view_fit(model_name)
  cal_data <- built$dat[seq_len(built$timesteps + 30), , drop = FALSE]
  # step_sequence() needs `timesteps` rows of history per prediction, so a
  # `new_data` slice must include at least that much leading context; this
  # slice yields exactly 5 rows with a full window.
  new_data <- cal_data[seq_len(built$timesteps + 4), , drop = FALSE]

  view_1 <- kerasnip_step_view(built$fit_obj, step = 1)
  conformal <- probably::int_conformal_split(view_1, cal_data = cal_data)
  result <- predict(conformal, new_data = new_data, level = 0.90)

  expect_s3_class(result, "tbl_df")
  expect_true(all(c(".pred", ".pred_lower", ".pred_upper") %in% names(result)))
  expect_equal(nrow(result), 5)
  expect_true(all(result$.pred_lower <= result$.pred_upper))
})

test_that("kerasnip_step_view: probably::int_conformal_full works per step", {
  skip_if_no_keras()
  skip_if_not_installed("probably")
  skip_if_not_installed("mgcv")

  model_name <- "step_view_conformal_full"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  # Kept small: int_conformal_full refits the whole model once per candidate
  # value of every new observation. `n` needs to be large enough that the
  # residual-variance model (an mgcv::gam()) sees a representative range of
  # `.pred` values during training — too narrow a range (small n) makes it
  # extrapolate wildly for new observations outside it, producing all-NA
  # bounds (this is what happened in CI with n = 40: see the analogous fix
  # in vignettes/multi_output_postprocessing.Rmd.orig).
  set.seed(42)
  n <- 150
  timesteps <- 5
  horizon <- 2
  dat <- tibble::tibble(value = sin(seq_len(n) / 10) + rnorm(n, sd = 0.05))

  rec <- recipe(dat) |>
    step_lead(value, lead = seq_len(horizon), prefix = "lead_") |>
    step_naomit(starts_with("lead_")) |>
    step_sequence(value, timesteps = timesteps, new_col = "window")

  input_block <- function(input_shape) {
    keras3::layer_input(shape = input_shape, name = "window_input")
  }
  lstm_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_lstm(units = units)
  }
  output_block <- function(tensor, units = 1) {
    tensor |> keras3::layer_dense(units = units)
  }

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      window = input_block,
      lstm = inp_spec(lstm_block, "window"),
      output = inp_spec(output_block, "lstm")
    ),
    mode = "regression"
  )

  spec <- get(model_name)(
    output_units = horizon,
    fit_epochs = 8,
    fit_verbose = 0
  ) |>
    set_engine("keras")
  wf <- workflow(rec, spec)

  split <- rsample::initial_time_split(dat, prop = 0.8)
  train_dat <- rsample::training(split)
  test_dat <- rsample::testing(split)

  fit_obj <- fit(wf, data = train_dat)
  view_1 <- kerasnip_step_view(fit_obj, step = 1)

  conformal <- probably::int_conformal_full(
    view_1,
    train_data = train_dat,
    control = probably::control_conformal_full(
      method = "grid",
      trial_points = 8
    )
  )
  new_data <- test_dat[seq_len(timesteps + 2), , drop = FALSE]
  result <- suppressWarnings(predict(
    conformal,
    new_data = new_data,
    level = 0.90
  ))

  expect_s3_class(result, "tbl_df")
  expect_true(all(c(".pred_lower", ".pred_upper") %in% names(result)))
  expect_equal(nrow(result), 3)
  expect_true(any(!is.na(result$.pred_lower)))
  expect_true(all(result$.pred_lower <= result$.pred_upper, na.rm = TRUE))
})

test_that("kerasnip_step_view: int_conformal_full errors when step_lead()/step_sequence() columns differ", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "step_view_conformal_full_mismatch"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  set.seed(1)
  n <- 40
  timesteps <- 5
  horizon <- 2
  dat <- tibble::tibble(
    value = sin(seq_len(n) / 10) + rnorm(n, sd = 0.05),
    other = rnorm(n)
  )

  rec <- recipe(dat) |>
    step_lead(value, lead = seq_len(horizon), prefix = "lead_") |>
    step_naomit(starts_with("lead_")) |>
    step_sequence(other, timesteps = timesteps, new_col = "window")

  input_block <- function(input_shape) {
    keras3::layer_input(shape = input_shape, name = "window_input")
  }
  lstm_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_lstm(units = units)
  }
  output_block <- function(tensor, units = 1) {
    tensor |> keras3::layer_dense(units = units)
  }

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      window = input_block,
      lstm = inp_spec(lstm_block, "window"),
      output = inp_spec(output_block, "lstm")
    ),
    mode = "regression"
  )

  spec <- get(model_name)(
    output_units = horizon,
    fit_epochs = 1,
    fit_verbose = 0
  ) |>
    set_engine("keras")
  fit_obj <- fit(workflow(rec, spec), data = dat)
  view_1 <- kerasnip_step_view(fit_obj, step = 1)

  expect_error(
    probably::int_conformal_full(view_1, train_data = dat),
    "single source column"
  )
})

test_that("kerasnip_step_view: int_conformal_full only supports control$method = \"grid\"", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "step_view_conformal_full_no_grid"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  built <- make_step_view_fit(
    model_name,
    n = 40,
    timesteps = 5,
    horizon = 2,
    fit_epochs = 1
  )
  view_1 <- kerasnip_step_view(built$fit_obj, step = 1)

  expect_error(
    probably::int_conformal_full(
      view_1,
      train_data = built$dat,
      control = probably::control_conformal_full(method = "iterative")
    ),
    "Only.*grid.*is supported"
  )
})

# =============================================================================
# Internal helpers: direct/mocked unit tests (no model training needed)
# =============================================================================

test_that("kerasnip_step_extract resolves the right column and errors when missing", {
  row_single_var <- tibble::tibble(.pred = 5)
  expect_equal(
    kerasnip:::kerasnip_step_extract(row_single_var, var = NULL, prefix = ".pred"),
    5
  )

  row_multi_var <- tibble::tibble(.pred_value = 7, .pred_other = 9)
  expect_equal(
    kerasnip:::kerasnip_step_extract(row_multi_var, var = "value", prefix = ".pred"),
    7
  )

  row_missing <- tibble::tibble(.pred_other = 9)
  expect_error(
    kerasnip:::kerasnip_step_extract(row_missing, var = "value", prefix = ".pred"),
    "Could not find"
  )
})

test_that("kerasnip_trial_fit_step_view returns an NA row when refit fails", {
  # A real but incomplete workflow (no preprocessor, no spec) genuinely
  # errors on fit(); no need to fake a failure to exercise the guard.
  view <- structure(
    list(workflow = workflows::workflow(), step = 1, var = NULL),
    class = "kerasnip_step_view"
  )
  trial_data <- tibble::tibble(value = c(1, 2, 3))
  res <- kerasnip:::kerasnip_trial_fit_step_view(
    trial = 5,
    trial_data = trial_data,
    view = view,
    level = 0.9,
    raw_col = "value",
    target_row_idx = 3,
    target_position = 1
  )
  expect_true(is.na(res$quantile))
  expect_true(is.na(res$.abs_resid))
})
