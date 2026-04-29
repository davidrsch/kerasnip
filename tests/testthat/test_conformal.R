# =============================================================================
# Conformal inference integration tests
#
# Verifies that all three conformal inference methods from the `probably`
# package work end-to-end with kerasnip workflows — for both the sequential
# and functional Keras APIs.
#
# All three methods are tested:
#   - int_conformal_split()  : calibration-set approach, no model refit
#   - int_conformal_cv()     : cross-validation approach, no model refit
#   - int_conformal_full()   : full conformal, refits model per candidate value
#
# int_conformal_full() is kept intentionally small (2 epochs, 5 rows of
# new_data, tight grid) to remain feasible in CI while still exercising
# the full code path.
# =============================================================================

# ---------------------------------------------------------------------------
# Shared sequential layer blocks
# ---------------------------------------------------------------------------
make_conformal_seq_blocks <- function() {
  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block <- function(model, units = 8) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }
  list(input = input_block, dense = dense_block, output = output_block)
}

# ---------------------------------------------------------------------------
# Shared functional layer blocks
# ---------------------------------------------------------------------------
make_conformal_func_blocks <- function() {
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

# ---------------------------------------------------------------------------
# Helper: assert that a conformal prediction result has the right shape
# ---------------------------------------------------------------------------
expect_valid_intervals <- function(result, n_rows) {
  expect_s3_class(result, "tbl_df")
  expect_true(
    all(c(".pred_lower", ".pred_upper") %in% names(result)),
    info = paste(
      "Expected .pred_lower and .pred_upper, got:",
      paste(names(result), collapse = ", ")
    )
  )
  expect_equal(nrow(result), n_rows)
  # Explicit NA checks: when int_conformal_full fails to determine bounds it
  # silently returns all-NA columns. all(NA <= NA) evaluates to NA, and
  # expect_true(NA) would pass — so we must check for NAs first.
  expect_false(
    anyNA(result$.pred_lower),
    label = ".pred_lower must not contain NA (did you forget fit_seed?)"
  )
  expect_false(
    anyNA(result$.pred_upper),
    label = ".pred_upper must not contain NA (did you forget fit_seed?)"
  )
  expect_true(all(result$.pred_lower <= result$.pred_upper))
}

# =============================================================================
# Sequential API
# =============================================================================

test_that("conformal: int_conformal_split works with sequential kerasnip workflow", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "conf_split_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_conformal_seq_blocks(),
    mode = "regression"
  )

  spec <- conf_split_seq(fit_epochs = 3) |> set_engine("keras")
  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  set.seed(42)
  split <- rsample::initial_split(data, prop = 0.70)
  train_dat <- rsample::training(split)
  cal_dat <- rsample::testing(split)

  fit_obj <- fit(wf, data = train_dat)
  conformal <- probably::int_conformal_split(fit_obj, cal_data = cal_dat)
  result <- predict(conformal, new_data = cal_dat[1:5, ], level = 0.90)

  expect_valid_intervals(result, 5)
})

test_that("conformal: int_conformal_cv works with sequential kerasnip workflow", {
  skip_if_no_keras()
  skip_if_not_installed("probably")
  skip_if_not_installed("tune")

  model_name <- "conf_cv_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_conformal_seq_blocks(),
    mode = "regression"
  )

  spec <- conf_cv_seq(fit_epochs = 2) |> set_engine("keras")
  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  set.seed(42)
  folds <- rsample::vfold_cv(data, v = 2)
  fitted_folds <- tune::fit_resamples(
    wf,
    resamples = folds,
    control = tune::control_resamples(
      save_pred = TRUE,
      extract = function(x) x
    )
  )
  conformal <- probably::int_conformal_cv(fitted_folds)
  result <- predict(conformal, new_data = data[1:5, ], level = 0.90)

  expect_valid_intervals(result, 5)
})

test_that("conformal: int_conformal_full works with sequential kerasnip workflow", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "conf_full_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_conformal_seq_blocks(),
    mode = "regression"
  )

  # fit_seed makes every internal refit deterministic, which is required for
  # probably's bounds-search to find monotone nonconformity scores. Without it
  # the intervals may be NA (and expect_valid_intervals will catch it).
  spec <- conf_full_seq(fit_epochs = 2, fit_seed = 42L) |> set_engine("keras")
  # Use a small subset to keep the number of model refits manageable in CI.
  # int_conformal_full refits the model for every candidate value of every
  # new test observation, so dataset and new_data size directly controls
  # total training cycles.
  data <- mtcars[1:15, ]
  new_data <- mtcars[16:17, ]
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  fit_obj <- fit(wf, data = data)
  conformal <- probably::int_conformal_full(
    fit_obj,
    train_data = data,
    control = probably::control_conformal_full(
      method = "grid",
      trial_points = 20
    )
  )
  result <- predict(conformal, new_data = new_data, level = 0.90)

  expect_valid_intervals(result, nrow(new_data))
})

# =============================================================================
# Functional API
# =============================================================================

test_that("conformal: int_conformal_split works with functional kerasnip workflow", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "conf_split_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_conformal_func_blocks(),
    mode = "regression"
  )

  spec <- conf_split_func(fit_epochs = 3) |> set_engine("keras")
  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  set.seed(42)
  split <- rsample::initial_split(data, prop = 0.70)
  train_dat <- rsample::training(split)
  cal_dat <- rsample::testing(split)

  fit_obj <- fit(wf, data = train_dat)
  conformal <- probably::int_conformal_split(fit_obj, cal_data = cal_dat)
  result <- predict(conformal, new_data = cal_dat[1:5, ], level = 0.90)

  expect_valid_intervals(result, 5)
})

test_that("conformal: int_conformal_cv works with functional kerasnip workflow", {
  skip_if_no_keras()
  skip_if_not_installed("probably")
  skip_if_not_installed("tune")

  model_name <- "conf_cv_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_conformal_func_blocks(),
    mode = "regression"
  )

  spec <- conf_cv_func(fit_epochs = 2) |> set_engine("keras")
  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  set.seed(42)
  folds <- rsample::vfold_cv(data, v = 2)
  fitted_folds <- tune::fit_resamples(
    wf,
    resamples = folds,
    control = tune::control_resamples(
      save_pred = TRUE,
      extract = function(x) x
    )
  )
  conformal <- probably::int_conformal_cv(fitted_folds)
  result <- predict(conformal, new_data = data[1:5, ], level = 0.90)

  expect_valid_intervals(result, 5)
})

test_that("conformal: int_conformal_full works with functional kerasnip workflow", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "conf_full_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_conformal_func_blocks(),
    mode = "regression"
  )

  # fit_seed: same reason as the sequential conformal_full test above.
  spec <- conf_full_func(fit_epochs = 2, fit_seed = 42L) |> set_engine("keras")
  data <- mtcars[1:15, ]
  new_data <- mtcars[16:17, ]
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  fit_obj <- fit(wf, data = data)
  conformal <- probably::int_conformal_full(
    fit_obj,
    train_data = data,
    control = probably::control_conformal_full(
      method = "grid",
      trial_points = 20
    )
  )
  result <- predict(conformal, new_data = new_data, level = 0.90)

  expect_valid_intervals(result, nrow(new_data))
})

test_that("conformal: int_conformal_full warns when fit_seed is absent (sequential)", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "conf_full_warn_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_conformal_seq_blocks(),
    mode = "regression"
  )

  spec <- conf_full_warn_seq(fit_epochs = 2) |> set_engine("keras")
  data <- mtcars[1:15, ]
  # Single test row: minimises the number of internal refits while still
  # exercising the warning code path.
  new_data <- mtcars[16, , drop = FALSE]
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  fit_obj <- fit(wf, data = data)
  conformal <- probably::int_conformal_full(
    fit_obj,
    train_data = data,
    control = probably::control_conformal_full(
      method = "grid",
      trial_points = 5 # tiny grid — just enough to trigger the code path
    )
  )

  # furrr re-signals warnings from parallel workers, so our warning fires once
  # per refit rather than once overall. expect_warning captures the first
  # occurrence and passes, but the remaining copies leak into the test output
  # and appear in CRAN checks. withCallingHandlers muffles all subsequent
  # occurrences of the same kerasnip warning after expect_warning has already
  # seen it, keeping the test output clean without hiding unrelated warnings.
  withCallingHandlers(
    expect_warning(
      predict(conformal, new_data = new_data, level = 0.90),
      regexp = "fit_seed",
      fixed = FALSE
    ),
    warning = function(w) {
      if (grepl("fit_seed", conditionMessage(w), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
})

test_that("conformal: int_conformal_full warns when fit_seed is absent (functional)", {
  skip_if_no_keras()
  skip_if_not_installed("probably")

  model_name <- "conf_full_warn_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_conformal_func_blocks(),
    mode = "regression"
  )

  spec <- conf_full_warn_func(fit_epochs = 2) |> set_engine("keras")
  data <- mtcars[1:15, ]
  new_data <- mtcars[16, , drop = FALSE]
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  fit_obj <- fit(wf, data = data)
  conformal <- probably::int_conformal_full(
    fit_obj,
    train_data = data,
    control = probably::control_conformal_full(
      method = "grid",
      trial_points = 5
    )
  )

  withCallingHandlers(
    expect_warning(
      predict(conformal, new_data = new_data, level = 0.90),
      regexp = "fit_seed",
      fixed = FALSE
    ),
    warning = function(w) {
      if (grepl("fit_seed", conditionMessage(w), fixed = TRUE)) {
        invokeRestart("muffleWarning")
      }
    }
  )
})
