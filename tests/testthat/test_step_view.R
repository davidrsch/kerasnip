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

make_step_view_fit <- function(model_name, n = 120, timesteps = 8, horizon = 3) {
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

  spec <- get(model_name)(output_units = horizon, fit_epochs = 5, fit_verbose = 0) |>
    set_engine("keras")

  wf <- workflow(rec, spec)
  fit_obj <- fit(wf, data = dat)

  list(fit_obj = fit_obj, dat = dat, timesteps = timesteps, horizon = horizon)
}

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

  expected <- vapply(raw_preds$.pred, function(tbl) tbl$.pred[tbl$.step == 2], numeric(1))
  expect_equal(view_preds$.pred, expected)
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

  tlr <- tailor::tailor() |> tailor::adjust_numeric_calibration(method = "linear")
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
