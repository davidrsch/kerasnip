# =============================================================================
# `tailor` / `probably` postprocessing integration tests (single-output)
#
# Verifies that kerasnip's standard `.pred` / `.pred_class` / `.pred_<level>`
# prediction tibbles work with `tailor`'s adjustments and with `probably`'s
# probability calibration, both used directly (fit()/predict() on a
# predictions tibble) and through a full `workflows::add_tailor()` workflow.
#
# Multi-output and multistep models are out of scope here: `tailor` requires
# a single outcome/estimate column pair (see issue #47 discussion), so they
# are handled separately.
# =============================================================================

make_tailor_class_blocks <- function() {
  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block <- function(model, units = 8) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(model, num_classes) {
    model |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }
  list(input = input_block, dense = dense_block, output = output_block)
}

make_tailor_reg_blocks <- function() {
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  dense_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(tensor) keras3::layer_dense(tensor, units = 1)
  list(
    main_input = input_block,
    dense = inp_spec(dense_block, "main_input"),
    output = inp_spec(output_block, "dense")
  )
}

# =============================================================================
# tailor::adjust_probability_threshold()
# =============================================================================

test_that("tailor: adjust_probability_threshold works on kerasnip classification output", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "tailor_thresh_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_tailor_class_blocks(),
    mode = "classification"
  )

  spec <- tailor_thresh_seq(fit_epochs = 2) |> set_engine("keras")
  data <- modeldata::two_class_dat
  rec <- recipe(Class ~ ., data = data)
  wf <- workflow(rec, spec)

  fit_obj <- fit(wf, data = data)
  new_data <- data[1:10, ]

  preds_class <- predict(fit_obj, new_data = new_data, type = "class")
  preds_prob <- predict(fit_obj, new_data = new_data, type = "prob")
  pred_data <- dplyr::bind_cols(Class = new_data$Class, preds_class, preds_prob)

  lvls <- levels(new_data$Class)

  # Threshold near 0 (tailor requires strictly between 1e-10 and
  # 0.9999999999): the event probability is essentially always >= this, so
  # every row must be classified as the first level, regardless of model
  # quality.
  tlr_lo <- tailor::tailor() |> tailor::adjust_probability_threshold(1e-9)
  tlr_lo_fit <- fit(
    tlr_lo,
    pred_data,
    outcome = Class,
    estimate = .pred_class,
    probabilities = c(.pred_Class1, .pred_Class2)
  )
  result_lo <- predict(tlr_lo_fit, pred_data)
  expect_true(all(result_lo$.pred_class == lvls[1]))

  # Threshold near 1: the event probability is (almost surely) never this
  # high, so every row must be classified as the second level.
  tlr_hi <- tailor::tailor() |>
    tailor::adjust_probability_threshold(0.999999999)
  tlr_hi_fit <- fit(
    tlr_hi,
    pred_data,
    outcome = Class,
    estimate = .pred_class,
    probabilities = c(.pred_Class1, .pred_Class2)
  )
  result_hi <- predict(tlr_hi_fit, pred_data)
  expect_true(all(result_hi$.pred_class == lvls[2]))
})

test_that("tailor: full classification workflow with add_tailor() fits and predicts correctly", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "tailor_wf_class_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_tailor_class_blocks(),
    mode = "classification"
  )

  spec <- tailor_wf_class_seq(fit_epochs = 2) |> set_engine("keras")
  data <- modeldata::two_class_dat
  rec <- recipe(Class ~ ., data = data)

  tlr <- tailor::tailor() |> tailor::adjust_probability_threshold(0.3)
  wf <- workflow(rec, spec) |> workflows::add_tailor(tlr)

  fit_obj <- fit(wf, data = data)
  result <- predict(fit_obj, new_data = data[1:10, ])

  expect_s3_class(result, "tbl_df")
  expect_true(".pred_class" %in% names(result))
  expect_equal(nrow(result), 10)
  expect_equal(levels(result$.pred_class), levels(data$Class))
})

# =============================================================================
# tailor::adjust_numeric_calibration()
# =============================================================================

test_that("tailor: adjust_numeric_calibration works on single-output kerasnip regression output", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "tailor_cal_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_tailor_reg_blocks(),
    mode = "regression"
  )

  spec <- tailor_cal_func(fit_epochs = 3) |> set_engine("keras")
  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  set.seed(42)
  split <- rsample::initial_split(data, prop = 0.70)
  train_dat <- rsample::training(split)
  cal_dat <- rsample::testing(split)

  fit_obj <- fit(wf, data = train_dat)

  cal_preds <- predict(fit_obj, new_data = cal_dat)
  cal_data <- dplyr::bind_cols(mpg = cal_dat$mpg, cal_preds)

  tlr <- tailor::tailor() |>
    tailor::adjust_numeric_calibration(method = "linear")
  tlr_fit <- fit(tlr, cal_data, outcome = mpg, estimate = .pred)

  new_preds <- predict(fit_obj, new_data = cal_dat)
  result <- predict(tlr_fit, new_preds)

  expect_s3_class(result, "tbl_df")
  expect_true(".pred" %in% names(result))
  expect_equal(nrow(result), nrow(cal_dat))
  expect_true(is.numeric(result$.pred))
})

test_that("tailor: full regression workflow with add_tailor() numeric calibration fits and predicts correctly", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "tailor_wf_reg_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block <- function(model, units = 8) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      input = input_block,
      dense = dense_block,
      output = output_block
    ),
    mode = "regression"
  )

  spec <- tailor_wf_reg_seq(fit_epochs = 3) |> set_engine("keras")
  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)

  set.seed(42)
  split <- rsample::initial_split(data, prop = 0.70)
  train_dat <- rsample::training(split)
  cal_dat <- rsample::testing(split)

  tlr <- tailor::tailor() |>
    tailor::adjust_numeric_calibration(method = "linear")
  wf <- workflow(rec, spec) |> workflows::add_tailor(tlr)

  fit_obj <- fit(wf, data = train_dat, data_calibration = cal_dat)
  result <- predict(fit_obj, new_data = cal_dat)

  expect_s3_class(result, "tbl_df")
  expect_true(".pred" %in% names(result))
  expect_equal(nrow(result), nrow(cal_dat))
  expect_true(is.numeric(result$.pred))
})

# =============================================================================
# probably calibration on kerasnip classification probabilities
# =============================================================================

test_that("probably: cal_estimate_isotonic + cal_apply calibrates kerasnip classification probabilities", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "probably_cal_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_tailor_class_blocks(),
    mode = "classification"
  )

  spec <- probably_cal_seq(fit_epochs = 2) |> set_engine("keras")
  data <- modeldata::two_class_dat
  rec <- recipe(Class ~ ., data = data)
  wf <- workflow(rec, spec)

  set.seed(42)
  split <- rsample::initial_split(data, prop = 0.70)
  train_dat <- rsample::training(split)
  cal_dat <- rsample::testing(split)

  fit_obj <- fit(wf, data = train_dat)

  cal_probs <- predict(fit_obj, new_data = cal_dat, type = "prob")
  cal_data <- dplyr::bind_cols(Class = cal_dat$Class, cal_probs)

  cal_model <- probably::cal_estimate_isotonic(
    cal_data,
    truth = Class,
    estimate = c(.pred_Class1, .pred_Class2)
  )

  new_probs <- predict(fit_obj, new_data = cal_dat, type = "prob")
  new_data_with_probs <- dplyr::bind_cols(Class = cal_dat$Class, new_probs)
  calibrated <- probably::cal_apply(new_data_with_probs, cal_model)

  expect_s3_class(calibrated, "tbl_df")
  expect_true(all(c(".pred_Class1", ".pred_Class2") %in% names(calibrated)))
  expect_equal(nrow(calibrated), nrow(cal_dat))
  expect_true(all(
    abs(rowSums(calibrated[c(".pred_Class1", ".pred_Class2")]) - 1) < 1e-5
  ))
})
