# =============================================================================
# kerasnip_output_view() tests
#
# `tailor`/`probably` require a single outcome/estimate column pair; a
# kerasnip multi-output model produces multiple `.pred_<output>` columns from
# multiple truth columns (the standard `parsnip::maybe_multivariate()`
# shape), which neither package can consume directly (see issue #47).
# `kerasnip_output_view()` presents one output of a multi-output fit as an
# ordinary single-output fit, so it can be calibrated/post-processed with the
# usual `tailor`/`probably` calls, one output at a time.
# =============================================================================

make_view_reg_blocks <- function() {
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

make_view_class_blocks <- function() {
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

make_view_reg_data <- function(n = 60) {
  set.seed(1)
  tibble::tibble(
    x1 = rnorm(n),
    x2 = rnorm(n),
    output_1 = x1 + rnorm(n, sd = 0.1),
    output_2 = 2 * x2 + rnorm(n, sd = 0.1)
  )
}

make_view_class_data <- function(n = 80) {
  set.seed(1)
  tibble::tibble(
    x1 = rnorm(n),
    x2 = rnorm(n),
    output_1 = factor(ifelse(x1 > 0, "yes", "no")),
    output_2 = factor(ifelse(x2 > 0, "yes", "no"))
  )
}

# =============================================================================
# Constructor validation
# =============================================================================

test_that("kerasnip_output_view: errors when x is not a fitted workflow", {
  skip_if_no_keras()

  expect_error(kerasnip_output_view(list(), "output_1"), "workflow")
  expect_error(
    kerasnip_output_view(workflows::workflow(), "output_1"),
    "workflow"
  )
})

test_that("kerasnip_output_view: errors on a single-output fit", {
  skip_if_no_keras()

  model_name <- "view_single_out"
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

  spec <- view_single_out(fit_epochs = 1) |> set_engine("keras")
  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)
  fit_obj <- fit(workflow(rec, spec), data = data)

  expect_error(
    kerasnip_output_view(fit_obj, "mpg"),
    "multi-output"
  )
})

test_that("kerasnip_output_view: errors on an unknown output name", {
  skip_if_no_keras()

  model_name <- "view_unknown_out"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_view_reg_blocks(),
    mode = "regression"
  )

  spec <- view_unknown_out(fit_epochs = 1) |> set_engine("keras")
  data <- make_view_reg_data()
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)
  fit_obj <- fit(workflow(rec, spec), data = data)

  expect_error(
    kerasnip_output_view(fit_obj, "output_3"),
    "output_3"
  )
})

# =============================================================================
# Regression: predict(), manual tailor calibration, probably conformal split
# =============================================================================

test_that("kerasnip_output_view: predict() matches the raw multi-output column", {
  skip_if_no_keras()

  model_name <- "view_reg_predict"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_view_reg_blocks(),
    mode = "regression"
  )

  spec <- view_reg_predict(fit_epochs = 5) |> set_engine("keras")
  data <- make_view_reg_data()
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)
  fit_obj <- fit(workflow(rec, spec), data = data)

  raw_preds <- predict(fit_obj, new_data = data[1:10, ])

  view_1 <- kerasnip_output_view(fit_obj, "output_1")
  view_preds <- predict(view_1, new_data = data[1:10, ])

  expect_s3_class(view_preds, "tbl_df")
  expect_equal(names(view_preds), ".pred")
  expect_equal(view_preds$.pred, raw_preds$.pred_output_1)

  # hardhat::extract_mold() slices the wrapped workflow's mold down to this
  # output's single column (what probably::int_conformal_split() relies on).
  mold <- hardhat::extract_mold(view_1)
  expect_equal(names(mold$outcomes), "output_1")

  # type = "conf_int" forwards through to the underlying Laplace intervals,
  # sliced down to this output like every other prediction type.
  raw_conf_int <- predict(fit_obj, new_data = data[1:5, ], type = "conf_int")
  view_conf_int <- predict(view_1, new_data = data[1:5, ], type = "conf_int")
  expect_equal(
    names(view_conf_int),
    c(".pred", ".pred_lower", ".pred_upper")
  )
  expect_equal(view_conf_int$.pred, raw_conf_int$.pred_output_1)
  expect_equal(view_conf_int$.pred_lower, raw_conf_int$.pred_lower_output_1)
  expect_equal(view_conf_int$.pred_upper, raw_conf_int$.pred_upper_output_1)
})

test_that("kerasnip_output_view: manual tailor::adjust_numeric_calibration works per output", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "view_reg_cal"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_view_reg_blocks(),
    mode = "regression"
  )

  spec <- view_reg_cal(fit_epochs = 5) |> set_engine("keras")
  data <- make_view_reg_data()
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)

  set.seed(42)
  split <- rsample::initial_split(data, prop = 0.7)
  train_dat <- rsample::training(split)
  cal_dat <- rsample::testing(split)

  fit_obj <- fit(workflow(rec, spec), data = train_dat)
  view_2 <- kerasnip_output_view(fit_obj, "output_2")

  cal_preds <- predict(view_2, new_data = cal_dat)
  cal_data <- dplyr::bind_cols(output_2 = cal_dat$output_2, cal_preds)

  tlr <- tailor::tailor() |>
    tailor::adjust_numeric_calibration(method = "linear")
  tlr_fit <- fit(tlr, cal_data, outcome = output_2, estimate = .pred)

  new_preds <- predict(view_2, new_data = cal_dat)
  result <- predict(tlr_fit, new_preds)

  expect_s3_class(result, "tbl_df")
  expect_true(".pred" %in% names(result))
  expect_equal(nrow(result), nrow(cal_dat))
  expect_true(is.numeric(result$.pred))
})

test_that("kerasnip_output_view: probably::int_conformal_split works per output", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "view_reg_conformal"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_view_reg_blocks(),
    mode = "regression"
  )

  spec <- view_reg_conformal(fit_epochs = 5) |> set_engine("keras")
  data <- make_view_reg_data()
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)

  set.seed(42)
  split <- rsample::initial_split(data, prop = 0.7)
  train_dat <- rsample::training(split)
  cal_dat <- rsample::testing(split)

  fit_obj <- fit(workflow(rec, spec), data = train_dat)
  view_1 <- kerasnip_output_view(fit_obj, "output_1")

  conformal <- probably::int_conformal_split(view_1, cal_data = cal_dat)
  result <- predict(conformal, new_data = cal_dat[1:5, ], level = 0.90)

  expect_s3_class(result, "tbl_df")
  expect_true(all(c(".pred", ".pred_lower", ".pred_upper") %in% names(result)))
  expect_equal(nrow(result), 5)
  expect_true(all(result$.pred_lower <= result$.pred_upper))
})

test_that("kerasnip_output_view: probably::int_conformal_full works per output", {
  skip_if_no_keras()
  skip_if_not_installed("probably")
  skip_if_not_installed("mgcv")

  model_name <- "view_reg_conformal_full"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_view_reg_blocks(),
    mode = "regression"
  )

  spec <- view_reg_conformal_full(fit_epochs = 5) |> set_engine("keras")
  # Kept small: int_conformal_full refits the whole multi-output model once
  # per candidate value of every new observation. `n` needs to be large
  # enough that the residual-variance model (an mgcv::gam()) sees a
  # representative range of `.pred` values during training — too narrow a
  # range (small n) makes it extrapolate wildly for new observations
  # outside it, producing all-NA bounds (this is what happened in CI with
  # n = 15: see the analogous fix in
  # vignettes/multi_output_postprocessing.Rmd.orig).
  data <- make_view_reg_data(n = 80)
  new_data <- make_view_reg_data(n = 82)[81:82, ]
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)

  fit_obj <- fit(workflow(rec, spec), data = data)
  view_1 <- kerasnip_output_view(fit_obj, "output_1")

  conformal <- probably::int_conformal_full(
    view_1,
    train_data = data,
    control = probably::control_conformal_full(
      method = "grid",
      trial_points = 10
    )
  )
  result <- suppressWarnings(predict(
    conformal,
    new_data = new_data,
    level = 0.90
  ))

  expect_s3_class(result, "tbl_df")
  expect_true(all(c(".pred_lower", ".pred_upper") %in% names(result)))
  expect_equal(nrow(result), nrow(new_data))
  expect_true(any(!is.na(result$.pred_lower)))
  expect_true(all(result$.pred_lower <= result$.pred_upper, na.rm = TRUE))

  # control defaults to method = "grid" when not supplied.
  conformal_default <- probably::int_conformal_full(view_1, train_data = data)
  expect_equal(conformal_default$control$method, "grid")

  # Any other method is rejected up front (before fitting the variance
  # model), since only "grid" is implemented for a multi-output view.
  expect_error(
    probably::int_conformal_full(
      view_1,
      train_data = data,
      control = probably::control_conformal_full()
    ),
    "grid"
  )
})

test_that("kerasnip_output_view: int_conformal_full errors for a classification view", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "view_class_conformal_full_guard"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_view_class_blocks(),
    mode = "classification"
  )

  spec <- view_class_conformal_full_guard(fit_epochs = 1) |> set_engine("keras")
  data <- make_view_class_data(n = 20)
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)
  fit_obj <- fit(workflow(rec, spec), data = data)
  view_1 <- kerasnip_output_view(fit_obj, "output_1")

  expect_error(
    probably::int_conformal_full(view_1, train_data = data),
    "regression"
  )
})

# =============================================================================
# Classification: predict(), manual tailor probability threshold
# =============================================================================

test_that("kerasnip_output_view: predict(type = 'prob'/'class') matches the raw multi-output columns", {
  skip_if_no_keras()

  model_name <- "view_class_predict"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_view_class_blocks(),
    mode = "classification"
  )

  spec <- view_class_predict(fit_epochs = 5) |> set_engine("keras")
  data <- make_view_class_data()
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)
  fit_obj <- fit(workflow(rec, spec), data = data)

  raw_class <- predict(fit_obj, new_data = data[1:10, ], type = "class")
  raw_prob <- predict(fit_obj, new_data = data[1:10, ], type = "prob")

  view_1 <- kerasnip_output_view(fit_obj, "output_1")
  view_class <- predict(view_1, new_data = data[1:10, ], type = "class")
  view_prob <- predict(view_1, new_data = data[1:10, ], type = "prob")

  expect_equal(names(view_class), ".pred_class")
  expect_equal(view_class$.pred_class, raw_class$.pred_class_output_1)

  expect_equal(sort(names(view_prob)), c(".pred_no", ".pred_yes"))
  expect_equal(view_prob$.pred_no, raw_prob$.pred_output_1_no)
  expect_equal(view_prob$.pred_yes, raw_prob$.pred_output_1_yes)

  # A view built (bypassing kerasnip_output_view()'s own name check) for an
  # output the real fit doesn't have: none of its real `.pred_<level>`
  # columns carry a matching `output_3` suffix.
  bad_view <- structure(
    list(workflow = fit_obj, output = "output_3", mode = "classification"),
    class = "kerasnip_output_view"
  )
  expect_error(
    predict(bad_view, new_data = data[1:5, ], type = "prob"),
    "Could not find probability columns"
  )
})

test_that("kerasnip_output_view: manual tailor::adjust_probability_threshold works per output", {
  skip_if_no_keras()
  skip_if_not_installed("tailor")

  model_name <- "view_class_thresh"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_view_class_blocks(),
    mode = "classification"
  )

  spec <- view_class_thresh(fit_epochs = 5) |> set_engine("keras")
  data <- make_view_class_data()
  rec <- recipe(output_1 + output_2 ~ x1 + x2, data = data)
  fit_obj <- fit(workflow(rec, spec), data = data)

  view_1 <- kerasnip_output_view(fit_obj, "output_1")
  new_data <- data[1:10, ]

  preds_class <- predict(view_1, new_data = new_data, type = "class")
  preds_prob <- predict(view_1, new_data = new_data, type = "prob")
  pred_data <- dplyr::bind_cols(
    output_1 = new_data$output_1,
    preds_class,
    preds_prob
  )

  lvls <- levels(new_data$output_1)

  tlr_lo <- tailor::tailor() |> tailor::adjust_probability_threshold(1e-9)
  tlr_lo_fit <- fit(
    tlr_lo,
    pred_data,
    outcome = output_1,
    estimate = .pred_class,
    probabilities = c(.pred_no, .pred_yes)
  )
  result_lo <- predict(tlr_lo_fit, pred_data)
  # `probabilities[1]` (.pred_no) is the reference column; a near-zero
  # threshold means everyone clears it.
  expect_true(all(result_lo$.pred_class == lvls[1]))
})

# =============================================================================
# Internal helpers: direct/mocked unit tests (no model training needed)
# =============================================================================

test_that("kerasnip_compute_bound warns and returns NA when no bound is found", {
  res <- tibble::tibble(trial = c(1, 2, 3), difference = c(-1, -2, -3))
  expect_warning(
    out <- kerasnip:::kerasnip_compute_bound(res, predicted = 2),
    "Could not determine bounds"
  )
  expect_true(is.na(out$.pred_lower))
  expect_true(is.na(out$.pred_upper))
})

test_that("kerasnip_trial_fit_output_view returns an NA row when refit fails", {
  # A real but incomplete workflow (no preprocessor, no spec) genuinely
  # errors on fit(); no need to fake a failure to exercise the guard.
  view <- structure(
    list(workflow = workflows::workflow(), output = "y"),
    class = "kerasnip_output_view"
  )
  trial_data <- tibble::tibble(y = c(1, 2, NA_real_))
  res <- kerasnip:::kerasnip_trial_fit_output_view(
    5,
    trial_data,
    view,
    0.9,
    "y"
  )
  expect_true(is.na(res$quantile))
  expect_true(is.na(res$.abs_resid))
})
