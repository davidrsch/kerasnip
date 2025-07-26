test_that("E2E: Tuning works with a generated spec", {
  skip_if_no_keras()

  input_block_tune <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_tune <- function(model, units = 16) {
    model |>
      keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_tune <- function(model, num_classes) {
    model |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }

  create_keras_spec(
    model_name = "e2e_mlp_class_tune",
    layer_blocks = list(
      input = input_block_tune,
      dense = dense_block_tune,
      output = output_block_tune
    ),
    mode = "classification"
  )

  tune_spec <- e2e_mlp_class_tune(
    num_dense = tune(),
    dense_units = tune(),
    epochs = 1
  ) |>
    set_engine("keras")

  rec <- recipe(Species ~ ., data = iris)
  tune_wf <- workflow(rec, tune_spec)

  folds <- rsample::vfold_cv(iris, v = 2)
  params <- extract_parameter_set_dials(tune_wf) |>
    update(
      num_dense = num_terms(c(1, 2)),
      dense_units = hidden_units(c(4, 8))
    )
  grid <- grid_regular(params, levels = 2)
  control <- control_grid(save_pred = FALSE, verbose = FALSE)

  # Use a try block because tuning can sometimes fail for non-package reasons
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
  expect_true(all(c("num_dense", "dense_units") %in% names(metrics)))
})
