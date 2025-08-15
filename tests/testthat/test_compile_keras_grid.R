# --- Test Data ---
x_train <- as.matrix(iris[, 1:4])
y_train <- iris$Species
# --- Tests ---

test_that("compile_keras_grid works for sequential models", {
  skip_on_cran()

  model_name <- "test_seq_spec_compile"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    mode = "classification",
    layer_blocks = list(
      dense = function(model, units = 32, activation = "relu") {
        if (is.null(model)) {
          keras3::keras_model_sequential(input_shape = 4) |>
            keras3::layer_dense(units = units, activation = activation)
        } else {
          model |> keras3::layer_dense(units = units, activation = activation)
        }
      },
      output = function(model, num_classes) {
        model |>
          keras3::layer_dense(units = num_classes, activation = "softmax")
      }
    )
  )

  spec <- test_seq_spec_compile() |>
    set_engine("keras")

  grid <- tibble::tibble(
    dense_units = c(16, 32),
    learn_rate = c(0.01, 0.001)
  )

  results <- compile_keras_grid(spec, grid, x_train, y_train)

  expect_s3_class(results, "tbl_df")
  expect_equal(nrow(results), 2)
  expect_true(all(
    c(
      "dense_units",
      "learn_rate",
      "compiled_model",
      "error"
    ) %in%
      names(results)
  ))
  expect_true(all(is.na(results$error)))
  expect_true(all(sapply(
    results$compiled_model,
    inherits,
    "keras.src.models.model.Model"
  )))
})

test_that("compile_keras_grid works for functional models", {
  skip_on_cran()

  model_name <- "test_func_spec_compile"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    mode = "classification",
    layer_blocks = list(
      input = function(input_shape) {
        keras3::layer_input(shape = input_shape)
      },
      dense = function(input, units = 32) {
        input |> keras3::layer_dense(units = units, activation = "relu")
      },
      output = function(dense, num_classes) {
        dense |>
          keras3::layer_dense(units = num_classes, activation = "softmax")
      }
    )
  )

  spec <- test_func_spec_compile() |>
    set_engine("keras")

  grid <- tibble::tibble(
    dense_units = c(16, 32),
    learn_rate = c(0.01, 0.001)
  )

  results <- compile_keras_grid(spec, grid, x_train, y_train)

  expect_s3_class(results, "tbl_df")
  expect_equal(nrow(results), 2)
  expect_true(all(
    c(
      "dense_units",
      "learn_rate",
      "compiled_model",
      "error"
    ) %in%
      names(results)
  ))
  expect_true(all(is.na(results$error)))
  expect_true(all(sapply(
    results$compiled_model,
    inherits,
    "keras.src.models.model.Model"
  )))
})

test_that("compile_keras_grid handles errors gracefully", {
  skip_on_cran()

  model_name <- "test_bad_func_spec_compile"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    mode = "classification",
    layer_blocks = list(
      input = function(input_shape) {
        keras3::layer_input(shape = input_shape)
      },
      dense1 = function(input, units = 32) {
        input |> keras3::layer_dense(units = units, activation = "relu")
      },
      dense2 = function(units = 16) {
        # Missing input tensor
        keras3::layer_dense(units = units, activation = "relu")
      },
      output = function(dense2, num_classes) {
        dense2 |>
          keras3::layer_dense(units = num_classes, activation = "softmax")
      }
    )
  )

  spec <- test_bad_func_spec_compile() |>
    set_engine("keras")

  grid <- tibble::tibble(dense1_units = 16)

  expect_warning(
    results <- compile_keras_grid(spec, grid, x_train, y_train),
    "Block 'dense2' has no inputs from other blocks."
  )

  expect_s3_class(results, "tbl_df")
  expect_equal(nrow(results), 1)
  expect_false(is.na(results$error[1]))
  expect_true(is.null(results$compiled_model[[1]]))
})
