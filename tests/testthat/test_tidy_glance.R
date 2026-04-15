test_that("tidy.kerasnip_model_fit returns correct structure", {
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
    model_name = "tidy_test_model",
    layer_blocks = list(
      input = input_block,
      hidden = hidden_block,
      output = output_block
    ),
    mode = "regression"
  )
  on.exit(remove_keras_spec("tidy_test_model"), add = TRUE)

  x_train <- matrix(rnorm(50 * 3), ncol = 3)
  y_train <- rnorm(50)
  train_df <- data.frame(x = I(x_train), y = y_train)

  spec <- tidy_test_model(fit_epochs = 2L) |>
    parsnip::set_engine("keras")
  fitted <- parsnip::fit(spec, y ~ x, data = train_df)

  result <- generics::tidy(fitted)

  expect_s3_class(result, "tbl_df")
  expect_named(result, c("layer", "class", "n_params"))
  expect_gt(nrow(result), 0L)
  expect_type(result$layer, "character")
  expect_type(result$class, "character")
  expect_type(result$n_params, "integer")
})

test_that("glance.kerasnip_model_fit returns one-row tibble with loss", {
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
    model_name = "glance_test_model",
    layer_blocks = list(
      input = input_block,
      hidden = hidden_block,
      output = output_block
    ),
    mode = "regression"
  )
  on.exit(remove_keras_spec("glance_test_model"), add = TRUE)

  x_train <- matrix(rnorm(50 * 3), ncol = 3)
  y_train <- rnorm(50)
  train_df <- data.frame(x = I(x_train), y = y_train)

  spec <- glance_test_model(fit_epochs = 2L) |>
    parsnip::set_engine("keras")
  fitted <- parsnip::fit(spec, y ~ x, data = train_df)

  result <- generics::glance(fitted)

  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 1L)
  expect_true("loss" %in% names(result))
})

test_that("glance returns empty tibble when history is NULL", {
  # Simulate a butchered fit where history was stripped
  mock_fit <- structure(
    list(fit = list(history = NULL)),
    class = c("kerasnip_model_fit", "model_fit")
  )
  result <- generics::glance(mock_fit)
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0L)
})
