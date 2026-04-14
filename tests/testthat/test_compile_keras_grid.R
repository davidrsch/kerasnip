# --- Test Data ---
x_train <- as.matrix(iris[, 1:4])
y_train <- iris$Species
train_df <- tibble(x = I(x_train), y = y_train)
# --- Tests ---

test_that("compile_keras_grid errors if spec is not a model_spec", {
  expect_error(
    compile_keras_grid("not_a_spec", data.frame(), NULL, NULL),
    "`spec` must be a `parsnip` model specification."
  )
})

test_that("compile_keras_grid errors if grid is not a data frame", {
  spec <- parsnip::rand_forest()
  expect_error(
    compile_keras_grid(spec, "not_a_df", NULL, NULL),
    "`grid` must be a data frame or tibble."
  )
})

test_that("compile_keras_grid errors if grid has zero rows", {
  spec <- parsnip::rand_forest()
  expect_error(
    compile_keras_grid(spec, tibble::tibble(), NULL, NULL),
    "`grid` must have at least one row"
  )
})

test_that("compile_keras_grid works with single-row empty grid (no tuning columns)", {
  skip_on_cran()

  model_name <- "test_seq_spec_compile_notuning"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    mode = "classification",
    layer_blocks = list(
      dense = function(model, units = 32) {
        if (is.null(model)) {
          keras3::keras_model_sequential(input_shape = 4L) |>
            keras3::layer_dense(units = units, activation = "relu")
        } else {
          model |> keras3::layer_dense(units = units, activation = "relu")
        }
      },
      output = function(model, num_classes) {
        model |>
          keras3::layer_dense(units = num_classes, activation = "softmax")
      }
    )
  )

  spec <- test_seq_spec_compile_notuning(dense_units = 16L) |>
    set_engine("keras")

  # tibble(.rows = 1L) = "build once with the spec's current args, no tuning"
  results <- compile_keras_grid(
    spec,
    tibble::tibble(.rows = 1L),
    select(train_df, x),
    select(train_df, y)
  )

  expect_s3_class(results, "tbl_df")
  expect_equal(nrow(results), 1L)
  expect_true("compiled_model" %in% names(results))
  expect_true("error" %in% names(results))
  expect_true(is.na(results$error[[1L]]))
  expect_true(inherits(
    results$compiled_model[[1L]],
    "keras.src.models.model.Model"
  ))
})

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

  results <- compile_keras_grid(
    spec,
    grid,
    select(train_df, x),
    select(train_df, y)
  )

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

  input_block <- function(input_shape) {
    keras3::layer_input(shape = input_shape, name = "x")
  }

  dense_block <- function(tensor, units = 32) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }

  output_block <- function(tensor, num_classes) {
    tensor |>
      keras3::layer_dense(
        units = num_classes,
        activation = "softmax",
        name = "y"
      )
  }

  create_keras_functional_spec(
    model_name = model_name,
    mode = "classification",
    layer_blocks = list(
      input = input_block,
      dense = inp_spec(dense_block, "input"),
      output = inp_spec(output_block, "dense")
    )
  )

  spec <- test_func_spec_compile() |>
    set_engine("keras")

  rec <- recipe(y ~ x, data = train_df) # Recipe for two outputs
  wf <- workflow() |>
    add_recipe(rec) |>
    add_model(spec)

  fit_obj <- fit(wf, data = train_df)

  grid <- tibble::tibble(
    dense_units = c(16, 32),
    learn_rate = c(0.01, 0.001)
  )

  results <- compile_keras_grid(
    spec,
    grid,
    select(train_df, x),
    select(train_df, y)
  )

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
        keras3::layer_input(shape = input_shape, name = "x")
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
    results <- compile_keras_grid(
      spec,
      grid,
      select(train_df, x),
      select(train_df, y)
    ),
    "Block 'dense2' has no inputs from other blocks."
  )

  expect_s3_class(results, "tbl_df")
  expect_equal(nrow(results), 1)
  expect_false(is.na(results$error[1]))
  expect_true(is.null(results$compiled_model[[1]]))
})
