# =============================================================================
# kerasnip_add_tailor() tests
#
# `workflows::add_tailor()` cannot attach to a kerasnip multi-output
# workflow (tailor::fit() needs a single outcome/estimate column pair; see
# issue #47). kerasnip_add_tailor() is a kerasnip-owned analogue: it trains
# the tailor against one named output via kerasnip_output_view() and splices
# the adjusted output back into the full multi-output prediction tibble.
# =============================================================================

make_add_tailor_reg_blocks <- function() {
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  dense_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_1 <- function(tensor) {
    keras3::layer_dense(tensor, units = 1, name = "output_1")
  }
  output_2 <- function(tensor) {
    keras3::layer_dense(tensor, units = 1, name = "output_2")
  }
  list(
    main_input = input_block,
    dense = inp_spec(dense_block, "main_input"),
    output_1 = inp_spec(output_1, "dense"),
    output_2 = inp_spec(output_2, "dense")
  )
}

make_add_tailor_reg_data <- function(n = 60) {
  set.seed(1)
  tibble::tibble(
    x1 = rnorm(n),
    x2 = rnorm(n),
    output_1 = x1 + rnorm(n, sd = 0.1),
    output_2 = 2 * x2 + rnorm(n, sd = 0.1)
  )
}

make_add_tailor_class_blocks <- function() {
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  dense_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_1 <- function(tensor, num_classes) {
    keras3::layer_dense(
      tensor,
      units = num_classes,
      activation = "softmax",
      name = "output_1"
    )
  }
  output_2 <- function(tensor, num_classes) {
    keras3::layer_dense(
      tensor,
      units = num_classes,
      activation = "softmax",
      name = "output_2"
    )
  }
  list(
    main_input = input_block,
    dense = inp_spec(dense_block, "main_input"),
    output_1 = inp_spec(output_1, "dense"),
    output_2 = inp_spec(output_2, "dense")
  )
}

make_add_tailor_class_data <- function(n = 80) {
  set.seed(1)
  tibble::tibble(
    x1 = rnorm(n),
    x2 = rnorm(n),
    output_1 = factor(ifelse(x1 > 0, "yes", "no")),
    output_2 = factor(ifelse(x2 > 0, "yes", "no"))
  )
}

test_that("kerasnip_add_tailor: numeric calibration on one output of a regression workflow", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "add_tailor_reg"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_add_tailor_reg_blocks(),
    mode = "regression"
  )

  spec <- add_tailor_reg(fit_epochs = 5) |> set_engine("keras")
  data <- make_add_tailor_reg_data()
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)
  wf <- workflow(rec, spec)

  set.seed(42)
  split <- rsample::initial_split(data, prop = 0.7)
  train_dat <- rsample::training(split)
  cal_dat <- rsample::testing(split)

  tlr <- tailor::tailor() |>
    tailor::adjust_numeric_calibration(method = "linear")
  tailored_wf <- kerasnip_add_tailor(wf, tlr, output = "output_2")

  fit_obj <- fit(tailored_wf, data = train_dat, data_calibration = cal_dat)
  result <- predict(fit_obj, new_data = cal_dat)

  expect_s3_class(result, "tbl_df")
  expect_true(all(c(".pred_output_1", ".pred_output_2") %in% names(result)))
  expect_equal(nrow(result), nrow(cal_dat))
  expect_true(is.numeric(result$.pred_output_1))
  expect_true(is.numeric(result$.pred_output_2))

  # output_1 must be untouched: it should exactly match a plain predict()
  # from the same underlying (uncalibrated) fit.
  raw_preds <- predict(fit_obj$fit_obj, new_data = cal_dat)
  expect_equal(result$.pred_output_1, raw_preds$.pred_output_1)
})

test_that("kerasnip_add_tailor: probability threshold on one output of a classification workflow", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "add_tailor_class"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_add_tailor_class_blocks(),
    mode = "classification"
  )

  spec <- add_tailor_class(fit_epochs = 5) |> set_engine("keras")
  data <- make_add_tailor_class_data()
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)
  wf <- workflow(rec, spec)

  tlr <- tailor::tailor() |> tailor::adjust_probability_threshold(1e-9)
  tailored_wf <- kerasnip_add_tailor(wf, tlr, output = "output_1")

  fit_obj <- fit(tailored_wf, data = data)
  result <- predict(fit_obj, new_data = data[1:10, ])

  expect_s3_class(result, "tbl_df")
  expect_true(".pred_class_output_1" %in% names(result))
  expect_true(".pred_class_output_2" %in% names(result))

  lvls <- levels(data$output_1)
  # near-zero threshold on output_1 means every row clears it.
  expect_true(all(result$.pred_class_output_1 == lvls[1]))

  # output_2 must be untouched.
  raw_class <- predict(fit_obj$fit_obj, new_data = data[1:10, ], type = "class")
  expect_equal(result$.pred_class_output_2, raw_class$.pred_class_output_2)
})

test_that("kerasnip_add_tailor: numeric calibration on one step of a multistep workflow", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "add_tailor_step"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  set.seed(42)
  n <- 120
  timesteps <- 8
  horizon <- 3
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

  spec <- add_tailor_step(
    output_units = horizon,
    fit_epochs = 5,
    fit_verbose = 0
  ) |>
    set_engine("keras")
  wf <- workflow(rec, spec)

  tlr <- tailor::tailor() |>
    tailor::adjust_numeric_calibration(method = "linear")
  tailored_wf <- kerasnip_add_tailor(wf, tlr, step = 2)

  fit_obj <- fit(tailored_wf, data = dat)
  new_data <- dat[seq_len(timesteps + 10), , drop = FALSE]
  result <- predict(fit_obj, new_data = new_data)

  expect_s3_class(result, "tbl_df")
  expect_equal(names(result), ".pred")
  expect_true(is.list(result$.pred))

  first_tbl <- result$.pred[[1]]
  expect_named(first_tbl, c(".step", ".pred"))
  expect_equal(first_tbl$.step, seq_len(horizon))

  # Steps 1 and 3 must be untouched: they should exactly match a plain
  # predict() from the same underlying (uncalibrated) fit.
  raw_preds <- predict(fit_obj$fit_obj, new_data = new_data)
  for (i in seq_along(result$.pred)) {
    expect_equal(
      result$.pred[[i]]$.pred[c(1, 3)],
      raw_preds$.pred[[i]]$.pred[c(1, 3)]
    )
  }
})
