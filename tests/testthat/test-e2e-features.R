test_that("E2E: Customizing main arguments works", {
  skip_if_no_keras()

  input_block_feat <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_feat <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_feat <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_sequential_spec(
    model_name = "e2e_mlp_feat",
    layer_blocks = list(
      input = input_block_feat,
      dense = dense_block_feat,
      output = output_block_feat
    ),
    mode = "regression"
  )

  # Main arguments (like compile_*) should be set in the spec function,
  # not in set_engine().
  spec <- e2e_mlp_feat(
    epochs = 2,
    compile_optimizer = "sgd",
    compile_loss = "mae",
    compile_metrics = c("mean_squared_error", "root_mean_squared_error")
  ) |>
    parsnip::set_engine("keras")

  # This should now run without the parsnip warning about removing arguments
  fit_obj <- NULL
  expect_no_warning(
    fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)
  )

  # Also verify the arguments were correctly used during compilation
  keras_model <- fit_obj$fit$fit
  compiled_loss <- keras_model$loss
  compiled_optimizer <- tolower(keras_model$optimizer$name)
  compiled_metrics <- sapply(
    keras_model$metrics[[2]]$metrics,
    function(m) {
      m$name
    }
  )

  # Keras might add suffixes or use different casings, so check flexibly
  expect_true(grepl("mae", compiled_loss))
  expect_true(grepl("sgd", tolower(compiled_optimizer)))
  expect_true("mean_squared_error" %in% compiled_metrics)
  expect_true("root_mean_squared_error" %in% compiled_metrics)
})

test_that("E2E: Customizing fit arguments works", {
  skip_if_no_keras()

  input_block_fit <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_fit <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_fit <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_sequential_spec(
    model_name = "e2e_mlp_fit",
    layer_blocks = list(
      input = input_block_fit,
      dense = dense_block_fit,
      output = output_block_fit
    ),
    mode = "regression"
  )

  # Fit arguments (like validation_split, callbacks) should be set in the
  # spec function, not in set_engine().
  spec <- e2e_mlp_fit(
    fit_validation_split = 0.2,
    fit_callbacks = list(keras3::callback_early_stopping(patience = 1)),
    fit_epochs = 3,
    compile_metrics = "mean_squared_error"
  ) |>
    parsnip::set_engine("keras")

  # This will run without error if the arguments are passed correctly
  fit_obj <- NULL
  expect_no_error(fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars))

  # Check that the callback was used (model should stop early)
  expect_lt(length(fit_obj$fit$history$metrics$loss), 5)
})

test_that("E2E: Setting num_blocks = 0 works", {
  skip_if_no_keras()

  input_block_zero <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_zero <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_zero <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_sequential_spec(
    model_name = "e2e_mlp_zero",
    layer_blocks = list(
      input = input_block_zero,
      dense = dense_block_zero,
      output = output_block_zero
    ),
    mode = "regression"
  )

  spec <- e2e_mlp_zero(num_dense = 0, epochs = 2) |>
    parsnip::set_engine("keras")
  # This should fit a model with only an input and output layer
  expect_no_error(parsnip::fit(spec, mpg ~ ., data = mtcars))
})

test_that("E2E: Error handling for reserved names works", {
  bad_blocks <- list(
    compile = function(model) model, # "compile" is a reserved name
    dense = function(model, u = 1) model |> keras3::layer_dense(units = u)
  )

  expect_error(
    create_keras_sequential_spec("bad_spec", bad_blocks),
    regexp = "`compile` and `optimizer` are protected names"
  )
})
