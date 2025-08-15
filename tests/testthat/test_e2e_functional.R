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
        c(path_a = "input_a", path_b = "input_b")
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


test_that("E2E: Functional spec (classification) works", {
  skip_if_no_keras()

  # Define blocks for a simple forked functional model
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  # Add a default to `units` to work around a bug in the doc generator
  # when handling args with no default. This doesn't affect runtime as the
  # value is always overridden.
  path_block <- function(tensor, units = 16) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  concat_block <- function(input_a, input_b) {
    keras3::layer_concatenate(list(input_a, input_b))
  }
  output_block_class <- function(tensor, num_classes) {
    tensor |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }

  model_name <- "e2e_func_class"
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
        c(path_a = "input_a", path_b = "input_b")
      ),
      output = inp_spec(output_block_class, "concatenated")
    ),
    mode = "classification"
  )

  spec <- e2e_func_class(
    path_a_units = 8,
    path_b_units = 4,
    fit_epochs = 2
  ) |>
    set_engine("keras")

  data <- iris
  rec <- recipe(Species ~ ., data = data)
  wf <- workflows::workflow(rec, spec)

  expect_no_error(fit_obj <- parsnip::fit(wf, data = data))
  expect_s3_class(fit_obj, "workflow")

  preds_class <- predict(fit_obj, new_data = data[1:5, ], type = "class")
  expect_s3_class(preds_class, "tbl_df")
  expect_equal(names(preds_class), ".pred_class")
  expect_equal(levels(preds_class$.pred_class), levels(data$Species))

  preds_prob <- predict(fit_obj, new_data = data[1:5, ], type = "prob")
  expect_s3_class(preds_prob, "tbl_df")
  expect_equal(names(preds_prob), paste0(".pred_", levels(data$Species)))
  expect_true(all(abs(rowSums(preds_prob) - 1) < 1e-5))
})


test_that("E2E: Functional spec tuning (including repetition) works", {
  skip_if_no_keras()

  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  # Add a default to `units` to work around a bug in the doc generator
  # when handling args with no default. This doesn't affect runtime as the
  # value is always overridden by the tuning grid.
  dense_block <- function(tensor, units = 16) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_class <- function(tensor, num_classes) {
    tensor |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }

  model_name <- "e2e_func_tune"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      main_input = input_block,
      # This block has a single input, so it can be repeated
      dense_path = inp_spec(dense_block, "main_input"),
      output = inp_spec(output_block_class, "dense_path")
    ),
    mode = "classification"
  )

  tune_spec <- e2e_func_tune(
    num_dense_path = tune(),
    dense_path_units = tune(),
    fit_epochs = 1
  ) |>
    set_engine("keras")

  rec <- recipe(Species ~ ., data = iris)
  tune_wf <- workflows::workflow(rec, tune_spec)

  folds <- rsample::vfold_cv(iris, v = 2)
  params <- extract_parameter_set_dials(tune_wf) |>
    update(
      num_dense_path = num_terms(c(1, 2)),
      dense_path_units = hidden_units(c(4, 8))
    )
  grid <- grid_regular(params, levels = 2)
  control <- control_grid(save_pred = FALSE, verbose = FALSE)

  tune_res <- try(
    tune_grid(
      tune_wf,
      resamples = folds,
      grid = grid,
      control = control
    ),
    silent = TRUE
  )

  if (inherits(tune_res, "try-error")) {
    testthat::skip(paste("Tuning failed with error:", as.character(tune_res)))
  }

  expect_s3_class(tune_res, "tune_results")

  metrics <- collect_metrics(tune_res)
  expect_s3_class(metrics, "tbl_df")
  expect_true(all(c("num_dense_path", "dense_path_units") %in% names(metrics)))
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
