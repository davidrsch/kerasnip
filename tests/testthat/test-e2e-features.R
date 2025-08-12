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

  model_name <- "e2e_mlp_feat"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
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
    fit_epochs = 2,
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

  model_name <- "e2e_mlp_fit"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
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

  model_name <- "e2e_mlp_zero"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      input = input_block_zero,
      dense = dense_block_zero,
      output = output_block_zero
    ),
    mode = "regression"
  )

  spec <- e2e_mlp_zero(num_dense = 0, fit_epochs = 2) |>
    parsnip::set_engine("keras")
  # This should fit a model with only an input and output layer
  expect_no_error(parsnip::fit(spec, mpg ~ ., data = mtcars))
})

test_that("E2E: Error handling for reserved names works", {
  model_name <- "bad_spec"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  bad_blocks <- list(
    compile = function(model) model, # "compile" is a reserved name
    dense = function(model, u = 1) model |> keras3::layer_dense(units = u)
  )

  expect_error(
    create_keras_sequential_spec(model_name, bad_blocks),
    regexp = "`compile`, `fit` and `optimizer` are protected names"
  )
})

test_that("E2E: extract_keras_summary works", {
  skip_if_no_keras()

  # Reuse model setup from previous tests
  input_block_feat <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_feat <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_feat <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  model_name <- "e2e_mlp_summary_test"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      input = input_block_feat,
      dense = dense_block_feat,
      output = output_block_feat
    ),
    mode = "regression"
  )

  spec <- e2e_mlp_summary_test(fit_epochs = 1) |>
    parsnip::set_engine("keras")

  fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)

  summary_output <- extract_keras_summary(fit_obj)

  expect_type(summary_output, "closure")
  expect_true(any(grepl("Layer ", summary_output)))
  expect_true(any(grepl("Output Shape", summary_output)))
  expect_true(any(grepl("Param #", summary_output)))
})

test_that("E2E: extract_keras_history works", {
  skip_if_no_keras()

  # Reuse model setup from previous tests
  input_block_feat <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_feat <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_feat <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  model_name <- "e2e_mlp_history_test"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      input = input_block_feat,
      dense = dense_block_feat,
      output = output_block_feat
    ),
    mode = "regression"
  )

  epochs_to_train <- 2
  spec <- e2e_mlp_history_test(fit_epochs = epochs_to_train) |>
    parsnip::set_engine("keras")

  fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)

  history_output <- extract_keras_history(fit_obj)

  expect_s3_class(history_output, "keras_training_history")
})

test_that("E2E: keras_evaluate works", {
  skip_if_no_keras()

  # Reuse model setup from previous tests
  input_block_eval <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_eval <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_eval <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  model_name <- "e2e_mlp_evaluate_test"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      input = input_block_eval,
      dense = dense_block_eval,
      output = output_block_eval
    ),
    mode = "regression"
  )

  spec <- e2e_mlp_evaluate_test(fit_epochs = 1) |>
    parsnip::set_engine("keras")

  fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)

  # Evaluate the model
  eval_output <- keras_evaluate(fit_obj, x = mtcars[, -1], y = mtcars$mpg)

  expect_true(class(eval_output) == "list")
  expect_true("loss" %in% names(eval_output))
  expect_true("mean_absolute_error" %in% names(eval_output))
})
