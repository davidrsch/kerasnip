test_that("E2E: Functional spec (regression) works", {
  skip_if_no_keras()

  # Define blocks for a simple forked functional model
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  path_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  concat_block <- function(input_a, input_b) {
    keras3::layer_concatenate(list(input_a, input_b))
  }
  output_block_reg <- function(tensor) keras3::layer_dense(tensor, units = 1)

  model_name <- "e2e_func_reg"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  # Create a spec with two parallel paths that are then concatenated
  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      main_input = input_block,
      path_a = inp_spec(path_block, "main_input"),
      path_b = inp_spec(path_block, "main_input"),
      concatenated = inp_spec(
        concat_block,
        c(input_a = "path_a", input_b = "path_b")
      ),
      output = inp_spec(output_block_reg, "concatenated")
    ),
    mode = "regression"
  )

  spec <- e2e_func_reg(
    path_a_units = 32,
    path_b_units = 16,
    fit_epochs = 2
  ) |>
    set_engine("keras")

  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflows::workflow(rec, spec)

  expect_no_error(fit_obj <- parsnip::fit(wf, data = data))
  expect_s3_class(fit_obj, "workflow")

  preds <- predict(fit_obj, new_data = data[1:5, ])
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_equal(nrow(preds), 5)
  expect_true(is.numeric(preds$.pred))
})

test_that("E2E: Functional regression works with named predictors in formula", {
  skip_if_no_keras()

  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  path_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  concat_block <- function(input_a, input_b) {
    keras3::layer_concatenate(list(input_a, input_b))
  }
  output_block_reg <- function(tensor) keras3::layer_dense(tensor, units = 1)

  model_name <- "e2e_func_reg_named"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      main_input = input_block,
      path_a = inp_spec(path_block, "main_input"),
      path_b = inp_spec(path_block, "main_input"),
      concatenated = inp_spec(
        concat_block,
        c(input_a = "path_a", input_b = "path_b")
      ),
      output = inp_spec(output_block_reg, "concatenated")
    ),
    mode = "regression"
  )

  spec <- e2e_func_reg_named(
    fit_epochs = 1
  ) |>
    set_engine("keras")

  data <- mtcars
  # Use named predictors to cover the x <- data[, x_names, drop = FALSE] line
  expect_no_error(
    fit_obj <- fit(spec, mpg ~ cyl + disp, data = data)
  )
  expect_s3_class(fit_obj, "model_fit")
})

test_that("E2E: Block repetition works for functional models", {
  skip_if_no_keras()

  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  dense_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(tensor) keras3::layer_dense(tensor, units = 1)

  model_name <- "e2e_func_repeat"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      main_input = input_block,
      dense_path = inp_spec(dense_block, "main_input"),
      output = inp_spec(output_block, "dense_path")
    ),
    mode = "regression"
  )

  # --- Test with 1 repetition ---
  spec_1 <- e2e_func_repeat(num_dense_path = 1, fit_epochs = 1) |>
    set_engine("keras")
  fit_1 <- fit(spec_1, mpg ~ ., data = mtcars)
  model_1_layers <- fit_1 |>
    extract_keras_model() |>
    pluck("layers")

  # Expect 3 layers: Input, Dense, Output
  expect_equal(length(model_1_layers), 3)

  # --- Test with 2 repetitions ---
  spec_2 <- e2e_func_repeat(num_dense_path = 2, fit_epochs = 1) |>
    set_engine("keras")
  fit_2 <- fit(spec_2, mpg ~ ., data = mtcars)
  model_2_layers <- fit_2 |>
    extract_keras_model() |>
    pluck("layers")
  # Expect 4 layers: Input, Dense, Dense, Output
  expect_equal(length(model_2_layers), 4)

  # --- Test with 0 repetitions ---
  spec_3 <- e2e_func_repeat(num_dense_path = 0, fit_epochs = 1) |>
    set_engine("keras")
  fit_3 <- fit(spec_3, mpg ~ ., data = mtcars)
  model_3_layers <- fit_3 |>
    extract_keras_model() |>
    pluck("layers")
  # Expect 2 layers: Input, Output
  expect_equal(length(model_3_layers), 2)
})

test_that("E2E: Multi-input, multi-output functional regression works", {
  skip_if_no_keras()
  options(kerasnip.show_removal_messages = FALSE)
  on.exit(options(kerasnip.show_removal_messages = TRUE), add = TRUE)

  # Define layer blocks
  input_block_1 <- function(input_shape) {
    keras3::layer_input(shape = input_shape, name = "input_1")
  }
  input_block_2 <- function(input_shape) {
    keras3::layer_input(shape = input_shape, name = "input_2")
  }
  dense_path <- function(tensor, units = 16) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  concat_block <- function(in_1, in_2) {
    keras3::layer_concatenate(list(in_1, in_2))
  }
  output_block_1 <- function(tensor) {
    keras3::layer_dense(tensor, units = 1, name = "output_1")
  }
  output_block_2 <- function(tensor) {
    keras3::layer_dense(tensor, units = 1, name = "output_2")
  }

  model_name <- "multi_in_out_reg"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      input_a = input_block_1,
      input_b = input_block_2,
      path_a = inp_spec(dense_path, "input_a"),
      path_b = inp_spec(dense_path, "input_b"),
      concatenated = inp_spec(
        concat_block,
        c(in_1 = "path_a", in_2 = "path_b")
      ),
      output_1 = inp_spec(output_block_1, "concatenated"),
      output_2 = inp_spec(output_block_2, "concatenated")
    ),
    mode = "regression"
  )

  spec <- multi_in_out_reg(fit_epochs = 2) |> set_engine("keras")

  # Prepare dummy data
  set.seed(123)
  x1 <- matrix(rnorm(100 * 5), ncol = 5)
  x2 <- matrix(rnorm(100 * 3), ncol = 3)
  y1 <- rnorm(100)
  y2 <- rnorm(100)

  train_df <- tibble::tibble(
    input_a = lapply(seq_len(nrow(x1)), function(i) x1[i, , drop = FALSE]),
    input_b = lapply(seq_len(nrow(x2)), function(i) x2[i, , drop = FALSE]),
    output_1 = y1,
    output_2 = y2
  )

  rec <- recipe(output_1 + output_2 ~ input_a + input_b, data = train_df)
  wf <- workflows::workflow(rec, spec)

  expect_no_error(fit_obj <- parsnip::fit(wf, data = train_df))

  new_data_df <- tibble::tibble(
    input_a = lapply(seq_len(5), function(i) matrix(rnorm(5), ncol = 5)),
    input_b = lapply(seq_len(5), function(i) matrix(rnorm(3), ncol = 3))
  )
  preds <- predict(fit_obj, new_data = new_data_df)

  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), c(".pred_output_1", ".pred_output_2"))
  expect_equal(nrow(preds), 5)
  expect_true(is.numeric(preds$.pred_output_1))
  expect_true(is.numeric(preds$.pred_output_2))
})
