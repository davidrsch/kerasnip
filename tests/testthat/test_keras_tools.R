test_that("keras_evaluate throws error for missing processing functions", {
  # Create a dummy fit object that is missing the processing functions
  dummy_fit_object <- list(
    fit = list(
      fit = "dummy_keras_model"
      # process_x and process_y are missing
    )
  )
  class(dummy_fit_object) <- "model_fit"

  expect_error(
    keras_evaluate(dummy_fit_object, x = mtcars[, -1], y = mtcars$mpg),
    "Could not find processing functions in the model fit object."
  )
})

test_that("keras_model_to_bytes returns raw vector for a valid model", {
  skip_if_no_keras()

  model <- keras3::keras_model_sequential(input_shape = 1L) |>
    keras3::layer_dense(units = 1L)
  keras3::compile(model, optimizer = "adam", loss = "mse")

  bytes <- keras_model_to_bytes(model)
  expect_type(bytes, "raw")
  expect_gt(length(bytes), 0L)
})

test_that("keras_model_from_bytes restores a valid model", {
  skip_if_no_keras()

  model <- keras3::keras_model_sequential(input_shape = 1L) |>
    keras3::layer_dense(units = 1L)
  keras3::compile(model, optimizer = "adam", loss = "mse")

  bytes <- keras_model_to_bytes(model)
  restored <- keras_model_from_bytes(bytes)

  expect_true(inherits(restored, "keras.src.models.sequential.Sequential"))
  x <- matrix(1:3, ncol = 1)
  expect_equal(dim(predict(restored, x)), c(3L, 1L))
})

test_that("get_keras_object returns instances not constructors", {
  skip_if_no_keras()

  loss_obj <- get_keras_object("mse", "loss")
  metric_obj <- get_keras_object("mean_squared_error", "metric")

  # Must be instances (Python objects), not the bare constructor functions
  expect_false(identical(loss_obj, keras3::loss_mean_squared_error))
  expect_false(identical(metric_obj, keras3::metric_mean_squared_error))

  # Must be serialisable (the original bug: class objects fail get_config)
  model <- keras3::keras_model_sequential(input_shape = 1L) |>
    keras3::layer_dense(units = 1L)
  keras3::compile(
    model,
    optimizer = "adam",
    loss = loss_obj,
    metrics = list(metric_obj)
  )
  expect_type(keras_model_to_bytes(model), "raw")
})

test_that("keras_model_to_bytes warns when save_model fails", {
  skip_if_no_keras()

  # Passing a plain R object (not a Keras model) forces save_model() to
  # throw, which exercises the warning path in keras_model_to_bytes()
  # independently of Keras version or compilation arguments.
  expect_warning(
    keras_model_to_bytes(1L),
    "Could not serialize"
  )
})

test_that("keras_evaluate with y=NULL skips y-processing and re-throws on bad xptr", {
  skip_if_no_keras()

  # Mock fit object: valid processing functions, invalid Python object, no bytes.
  # This exercises:
  #   - line 98: y_proc <- NULL  (y = NULL path taken)
  #   - line 99: if (!is.null(y)) FALSE  (y branch skipped)
  #   - line 118: stop(e)  (xptr invalid, no keras_bytes to restore from)
  mock_fit <- list(
    fit = list(
      fit = 1L,
      keras_bytes = NULL,
      process_x = function(x, ...) list(x_proc = as.matrix(x)),
      process_y = function(y, ...) list(y_proc = as.integer(y))
    )
  )
  class(mock_fit) <- "model_fit"

  x_new <- as.data.frame(matrix(rnorm(4), nrow = 1))

  expect_error(keras_evaluate(mock_fit, x_new, y = NULL))
})
