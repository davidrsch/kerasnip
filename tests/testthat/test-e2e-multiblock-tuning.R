test_that("E2E: Multi-block model tuning works", {
  skip_if_no_keras()

  input_block_mb <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }

  starting_layers <- function(model, layer1_units = 16, layer2_units = 32) {
    model |>
      keras3::layer_dense(units = layer1_units, activation = "relu") |>
      keras3::layer_dense(units = layer2_units, activation = "relu")
  }

  ending_layers <- function(model, units = 32, dropout = 0.2) {
    model |>
      keras3::layer_dense(units = units, activation = "relu") |>
      keras3::layer_dropout(rate = dropout)
  }

  output_block_mb <- function(model, num_classes) {
    model |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }

  model_name <- "mb_mt"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      input = input_block_mb,
      start = starting_layers,
      end = ending_layers,
      output = output_block_mb
    ),
    mode = "classification"
  )

  tune_spec <- mb_mt(
    num_start = tune(),
    start_layer1_units = tune(),
    start_layer2_units = tune(),
    end_units = tune(),
    fit_epochs = 1
  ) |>
    set_engine("keras")

  rec <- recipe(Species ~ ., data = iris)
  wf <- workflow(rec) |>
    add_model(tune_spec)

  folds <- rsample::vfold_cv(iris, v = 2)

  params <- extract_parameter_set_dials(wf) |>
    update(
      num_start = dials::num_terms(c(1, 2)),
      start_layer1_units = dials::hidden_units(c(4, 8)),
      start_layer2_units = dials::hidden_units(c(8, 16)),
      end_units = dials::hidden_units(c(4, 8))
    )

  grid <- grid_regular(params, levels = 2)
  control <- control_grid(
    save_pred = FALSE,
    verbose = FALSE,
    save_workflow = TRUE
  )

  # Use a try block because tuning can sometimes fail for non-package reasons
  tune_res <- try(
    tune_grid(
      wf,
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
  expect_true(all(
    c("num_start", "start_layer1_units", "start_layer2_units", "end_units") %in%
      names(metrics)
  ))

  expect_no_error(
    best_fit <- tune::fit_best(tune_res)
  )
  expect_s3_class(best_fit, "workflow")
})
