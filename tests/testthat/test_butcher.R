test_that("axe_data strips history and predict still works", {
  skip_if_not_installed("butcher")
  skip_if_not(
    reticulate::py_module_available("keras"),
    "keras not available"
  )

  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  hidden_block <- function(model, units = 8) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(model, num_classes) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_sequential_spec(
    model_name = "butcher_test_model",
    layer_blocks = list(
      input = input_block,
      hidden = hidden_block,
      output = output_block
    ),
    mode = "regression"
  )
  on.exit(remove_keras_spec("butcher_test_model"), add = TRUE)

  x_train <- matrix(rnorm(50 * 3), ncol = 3)
  y_train <- rnorm(50)
  train_df <- data.frame(x = I(x_train), y = y_train)

  spec <- butcher_test_model(fit_epochs = 2L) |>
    parsnip::set_engine("keras")
  fitted <- parsnip::fit(spec, y ~ x, data = train_df)

  # History exists before axing
  expect_false(is.null(fitted$fit$history))

  axed <- butcher::axe_data(fitted)

  # History is gone
  expect_null(axed$fit$history)

  # predict still works
  new_df <- data.frame(x = I(matrix(rnorm(5 * 3), ncol = 3)))
  preds <- predict(axed, new_data = new_df)
  expect_s3_class(preds, "tbl_df")
  expect_equal(nrow(preds), 5L)
})

test_that("all axe_* methods are callable without error", {
  skip_if_not_installed("butcher")
  skip_if_not(
    reticulate::py_module_available("keras"),
    "keras not available"
  )

  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  output_block <- function(model, num_classes) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_sequential_spec(
    model_name = "butcher_noop_model",
    layer_blocks = list(
      input = input_block,
      output = output_block
    ),
    mode = "regression"
  )
  on.exit(remove_keras_spec("butcher_noop_model"), add = TRUE)

  x_train <- matrix(rnorm(50 * 3), ncol = 3)
  y_train <- rnorm(50)
  train_df <- data.frame(x = I(x_train), y = y_train)

  spec <- butcher_noop_model(fit_epochs = 2L) |>
    parsnip::set_engine("keras")
  fitted <- parsnip::fit(spec, y ~ x, data = train_df)

  expect_no_error(butcher::axe_env(fitted))
  expect_no_error(butcher::axe_call(fitted))
  expect_no_error(butcher::axe_ctrl(fitted))
  expect_no_error(butcher::axe_fitted(fitted))
})

test_that("axe_env no-op preserves predict functionality", {
  skip_if_not_installed("butcher")
  skip_if_not(
    reticulate::py_module_available("keras"),
    "keras not available"
  )

  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  output_block <- function(model, num_classes) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_sequential_spec(
    model_name = "butcher_env_model",
    layer_blocks = list(
      input = input_block,
      output = output_block
    ),
    mode = "regression"
  )
  on.exit(remove_keras_spec("butcher_env_model"), add = TRUE)

  x_train <- matrix(rnorm(50 * 3), ncol = 3)
  y_train <- rnorm(50)
  train_df <- data.frame(x = I(x_train), y = y_train)

  spec <- butcher_env_model(fit_epochs = 2L) |>
    parsnip::set_engine("keras")
  fitted <- parsnip::fit(spec, y ~ x, data = train_df)

  axed <- butcher::axe_env(fitted)
  new_df <- data.frame(x = I(matrix(rnorm(5 * 3), ncol = 3)))
  preds <- predict(axed, new_data = new_df)
  expect_s3_class(preds, "tbl_df")
  expect_equal(nrow(preds), 5L)
})
