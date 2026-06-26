# =============================================================================
# Last-Layer Laplace Approximation interval tests
#
# Verifies that predict(..., type = "conf_int") and
# predict(..., type = "pred_int") work end-to-end with kerasnip regression
# models — for both the sequential and functional Keras APIs.
#
# Tests cover:
#   - Single-output sequential and functional API models
#   - Column name correctness (.pred, .pred_lower, .pred_upper)
#   - Interval validity (lower <= upper)
#   - pred_int wider than conf_int (observation noise added)
#   - level argument respected
#   - Classification models error gracefully
#   - saveRDS/readRDS round-trip
# =============================================================================

# ---------------------------------------------------------------------------
# Shared sequential layer blocks (regression)
# ---------------------------------------------------------------------------
make_lla_seq_blocks <- function() {
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
# Shared functional layer blocks (regression)
# ---------------------------------------------------------------------------
make_lla_func_blocks <- function() {
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
# Helper: assert that interval predictions have the right shape
# ---------------------------------------------------------------------------
expect_valid_intervals <- function(result, n_rows, prefix = "") {
  testthat::expect_s3_class(result, "tbl_df")
  lower_col <- paste0(".pred_lower", prefix)
  upper_col <- paste0(".pred_upper", prefix)
  testthat::expect_true(
    all(c(lower_col, upper_col) %in% names(result)),
    info = paste(
      "Expected",
      lower_col,
      "and",
      upper_col,
      "in names, got:",
      paste(names(result), collapse = ", ")
    )
  )
  testthat::expect_equal(nrow(result), n_rows)
  testthat::expect_true(all(result[[lower_col]] <= result[[upper_col]]))
}

# =============================================================================
# Sequential API
# =============================================================================

test_that("LLA: sequential conf_int returns valid intervals", {
  skip_if_no_keras()

  model_name <- "lla_ci_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_seq_blocks(),
    mode = "regression"
  )

  spec <- lla_ci_seq(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)
  result <- predict(fit_obj, mtcars[1:5, ], type = "conf_int")

  expect_valid_intervals(result, 5)
  expect_true(".pred" %in% names(result))
})

test_that("LLA: sequential pred_int returns valid intervals", {
  skip_if_no_keras()

  model_name <- "lla_pi_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_seq_blocks(),
    mode = "regression"
  )

  spec <- lla_pi_seq(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)
  result <- predict(fit_obj, mtcars[1:5, ], type = "pred_int")

  expect_valid_intervals(result, 5)
  expect_true(".pred" %in% names(result))
})

test_that("LLA: pred_int intervals are wider than conf_int", {
  skip_if_no_keras()

  model_name <- "lla_wider_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_seq_blocks(),
    mode = "regression"
  )

  spec <- lla_wider_seq(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)

  ci <- predict(fit_obj, mtcars[1:5, ], type = "conf_int")
  pi <- predict(fit_obj, mtcars[1:5, ], type = "pred_int")

  # pred_int lower should be <= conf_int lower
  expect_true(all(pi$.pred_lower <= ci$.pred_lower))
  # pred_int upper should be >= conf_int upper
  expect_true(all(pi$.pred_upper >= ci$.pred_upper))
})

test_that("LLA: level argument affects interval width", {
  skip_if_no_keras()

  model_name <- "lla_level_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_seq_blocks(),
    mode = "regression"
  )

  spec <- lla_level_seq(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)

  ci90 <- predict(fit_obj, mtcars[1:5, ], type = "conf_int", level = 0.90)
  ci99 <- predict(fit_obj, mtcars[1:5, ], type = "conf_int", level = 0.99)

  width90 <- ci90$.pred_upper - ci90$.pred_lower
  width99 <- ci99$.pred_upper - ci99$.pred_lower
  expect_true(all(width90 < width99))
})

test_that("LLA: default level is 0.95", {
  skip_if_no_keras()

  model_name <- "lla_deflvl_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_seq_blocks(),
    mode = "regression"
  )

  spec <- lla_deflvl_seq(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)

  # Omitting level should default to 0.95
  ci_default <- predict(fit_obj, mtcars[1:5, ], type = "conf_int")
  ci_95 <- predict(fit_obj, mtcars[1:5, ], type = "conf_int", level = 0.95)

  expect_equal(ci_default, ci_95)
})

# =============================================================================
# Functional API
# =============================================================================

test_that("LLA: functional conf_int returns valid intervals", {
  skip_if_no_keras()

  model_name <- "lla_ci_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_lla_func_blocks(),
    mode = "regression"
  )

  spec <- lla_ci_func(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)
  result <- predict(fit_obj, mtcars[1:5, ], type = "conf_int")

  expect_valid_intervals(result, 5)
})

test_that("LLA: functional pred_int returns valid intervals", {
  skip_if_no_keras()

  model_name <- "lla_pi_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_lla_func_blocks(),
    mode = "regression"
  )

  spec <- lla_pi_func(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)
  result <- predict(fit_obj, mtcars[1:5, ], type = "pred_int")

  expect_valid_intervals(result, 5)
})

test_that("LLA: functional conf_int and pred_int work with level", {
  skip_if_no_keras()

  model_name <- "lla_lvl_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_lla_func_blocks(),
    mode = "regression"
  )

  spec <- lla_lvl_func(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)

  ci <- predict(fit_obj, mtcars[1:5, ], type = "conf_int", level = 0.90)
  pi <- predict(fit_obj, mtcars[1:5, ], type = "pred_int", level = 0.90)

  expect_valid_intervals(ci, 5)
  expect_valid_intervals(pi, 5)
  expect_true(all(pi$.pred_lower <= ci$.pred_lower))
  expect_true(all(pi$.pred_upper >= ci$.pred_upper))
})

# =============================================================================
# Classification — conf_int now works (as of classification LLA support)
# =============================================================================

test_that("LLA: classification conf_int returns per-class columns", {
  skip_if_no_keras()

  model_name <- "lla_cls_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  cls_blocks <- list(
    input = function(model, input_shape) {
      keras3::keras_model_sequential(input_shape = input_shape)
    },
    hidden = function(model) {
      model |> keras3::layer_dense(units = 8, activation = "relu")
    },
    output = function(model, num_classes) {
      model |>
        keras3::layer_dense(units = num_classes, activation = "softmax")
    }
  )

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = cls_blocks,
    mode = "classification"
  )

  spec <- lla_cls_seq(fit_epochs = 3) |> set_engine("keras")
  rec <- recipe(Species ~ ., iris)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris)
  result <- predict(fit_obj, iris[1:5, ], type = "conf_int")

  # Per-class columns on probability scale
  for (cl in levels(iris$Species)) {
    lo_col <- paste0(".pred_lower_", cl)
    hi_col <- paste0(".pred_upper_", cl)
    testthat::expect_true(lo_col %in% names(result))
    testthat::expect_true(hi_col %in% names(result))
    testthat::expect_true(all(result[[lo_col]] >= 0))
    testthat::expect_true(all(result[[lo_col]] <= 1))
    testthat::expect_true(all(result[[hi_col]] >= 0))
    testthat::expect_true(all(result[[hi_col]] <= 1))
    testthat::expect_true(all(result[[lo_col]] <= result[[hi_col]]))
  }
})

# =============================================================================
# saveRDS / readRDS round-trip
# =============================================================================

test_that("LLA: intervals survive saveRDS/readRDS round-trip", {
  skip_if_no_keras()

  model_name <- "lla_rds_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_seq_blocks(),
    mode = "regression"
  )

  spec <- lla_rds_seq(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)

  # Predict before round-trip
  ci_before <- predict(fit_obj, mtcars[1:5, ], type = "conf_int")

  # Save and reload
  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(fit_obj, tmp)
  rm(fit_obj)
  fit_restored <- readRDS(tmp)

  # Predict after round-trip
  ci_after <- predict(fit_restored, mtcars[1:5, ], type = "conf_int")

  expect_valid_intervals(ci_after, 5)
  expect_equal(ci_before, ci_after)
})

# =============================================================================
# Edge case: model with no hidden layers (input -> output only)
# =============================================================================

test_that("LLA: minimal model (no hidden layers) errors clearly", {
  skip_if_no_keras()

  model_name <- "lla_min_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  minimal_blocks <- list(
    input = function(model, input_shape) {
      keras3::keras_model_sequential(input_shape = input_shape)
    },
    output = function(model) {
      model |> keras3::layer_dense(units = 1)
    }
  )

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = minimal_blocks,
    mode = "regression"
  )

  spec <- lla_min_seq(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(mpg ~ ., mtcars)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, mtcars)

  # No hidden layers before the output Dense -> Laplace unavailable
  expect_error(
    predict(fit_obj, mtcars[1:5, ], type = "conf_int"),
    "Laplace confidence intervals are not available"
  )
})

# =============================================================================
# Multi-output functional regression
# =============================================================================

test_that("LLA: multi-output functional regression conf_int works", {
  skip_if_no_keras()

  model_name <- "lla_multi_reg"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  dense_block <- function(tensor, units = 6) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  out1_block <- function(tensor) {
    keras3::layer_dense(tensor, units = 1, name = "y1")
  }
  out2_block <- function(tensor) {
    keras3::layer_dense(tensor, units = 1, name = "y2")
  }

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      main_input = input_block,
      shared = inp_spec(dense_block, "main_input"),
      y1 = inp_spec(out1_block, "shared"),
      y2 = inp_spec(out2_block, "shared")
    ),
    mode = "regression"
  )

  set.seed(123)
  n <- 100
  train_df <- tibble::tibble(
    x1 = rnorm(n),
    x2 = rnorm(n),
    y1 = rnorm(n),
    y2 = rnorm(n)
  )

  spec <- lla_multi_reg(fit_epochs = 5) |> set_engine("keras")
  rec <- recipe(y1 + y2 ~ x1 + x2, train_df)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, train_df)
  result <- predict(fit_obj, train_df[1:5, ], type = "conf_int")

  expect_s3_class(result, "tbl_df")
  expect_true(".pred_lower_y1" %in% names(result))
  expect_true(".pred_upper_y1" %in% names(result))
  expect_true(".pred_lower_y2" %in% names(result))
  expect_true(".pred_upper_y2" %in% names(result))
  expect_equal(nrow(result), 5)
})
