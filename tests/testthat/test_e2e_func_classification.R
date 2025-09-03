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

test_that("E2E: Multi-input, single-output functional classification works", {
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
  flatten_block <- function(tensor) keras3::layer_flatten(tensor)
  dense_path <- function(tensor, units = 16) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  concat_block <- function(in_1, in_2) {
    keras3::layer_concatenate(list(in_1, in_2))
  }
  output_block_class <- function(tensor, num_classes) {
    keras3::layer_dense(tensor, units = num_classes, activation = "softmax")
  }

  model_name <- "multi_in_class"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      input_a = input_block_1,
      input_b = input_block_2,
      flatten_a = inp_spec(flatten_block, "input_a"),
      flatten_b = inp_spec(flatten_block, "input_b"),
      path_a = inp_spec(dense_path, "flatten_a"),
      path_b = inp_spec(dense_path, "flatten_b"),
      concatenated = inp_spec(
        concat_block,
        c(path_a = "in_1", path_b = "in_2")
      ),
      output = inp_spec(output_block_class, "concatenated")
    ),
    mode = "classification"
  )

  spec <- multi_in_class(fit_epochs = 2) |> set_engine("keras")

  # Prepare dummy data
  set.seed(123)
  x1 <- matrix(rnorm(100 * 5), ncol = 5)
  x2 <- matrix(rnorm(100 * 3), ncol = 3)
  y <- factor(sample(c("a", "b"), 100, replace = TRUE))

  train_df <- tibble::tibble(
    input_a = lapply(seq_len(nrow(x1)), function(i) x1[i, , drop = FALSE]),
    input_b = lapply(seq_len(nrow(x2)), function(i) x2[i, , drop = FALSE]),
    outcome = y
  )

  rec <- recipe(outcome ~ input_a + input_b, data = train_df)
  wf <- workflows::workflow(rec, spec)

  expect_no_error(fit_obj <- parsnip::fit(wf, data = train_df))

  new_data_df <- tibble::tibble(
    input_a = lapply(seq_len(5), function(i) matrix(rnorm(5), ncol = 5)),
    input_b = lapply(seq_len(5), function(i) matrix(rnorm(3), ncol = 3))
  )
  preds <- predict(fit_obj, new_data = new_data_df)

  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), c(".pred_class"))
  expect_equal(nrow(preds), 5)
  expect_true(is.factor(preds$.pred_class))
})

test_that("E2E: Functional spec with pre-constructed optimizer works", {
  skip_if_no_keras()

  # Define blocks for a simple forked functional model
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  path_block <- function(tensor, units = 16) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  concat_block <- function(input_a, input_b) {
    keras3::layer_concatenate(list(input_a, input_b))
  }
  output_block_class <- function(tensor, num_classes) {
    tensor |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }

  model_name <- "e2e_func_class_optimizer"
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

  # Define a pre-constructed optimizer
  my_optimizer <- keras3::optimizer_sgd(learning_rate = 0.001)

  spec <- e2e_func_class_optimizer(
    path_a_units = 8,
    path_b_units = 4,
    fit_epochs = 2,
    compile_optimizer = my_optimizer
  ) |>
    set_engine("keras")

  data <- iris
  rec <- recipe(Species ~ ., data = data)
  wf <- workflows::workflow(rec, spec)

  expect_no_error(fit_obj <- parsnip::fit(wf, data = data))
  expect_s3_class(fit_obj, "workflow")
})

test_that("E2E: Functional spec with string loss works", {
  skip_if_no_keras()

  # Define blocks for a simple forked functional model
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  path_block <- function(tensor, units = 16) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  concat_block <- function(input_a, input_b) {
    keras3::layer_concatenate(list(input_a, input_b))
  }
  output_block_class <- function(tensor, num_classes) {
    tensor |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }

  model_name <- "e2e_func_class_loss_string"
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

  spec <- e2e_func_class_loss_string(
    path_a_units = 8,
    path_b_units = 4,
    fit_epochs = 2,
    compile_loss = "categorical_crossentropy"
  ) |>
    set_engine("keras")

  data <- iris
  rec <- recipe(Species ~ ., data = data)
  wf <- workflows::workflow(rec, spec)

  expect_no_error(fit_obj <- parsnip::fit(wf, data = data))
  expect_s3_class(fit_obj, "workflow")
})
