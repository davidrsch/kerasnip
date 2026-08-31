test_that("stacks ensemble works end-to-end with kerasnip models (#48)", {
  skip_if_not_installed("stacks")
  skip_if_not(
    reticulate::py_module_available("keras"),
    "keras not available"
  )

  model_name <- "stacks_mlp"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      input = function(model, input_shape) {
        keras3::keras_model_sequential(input_shape = input_shape)
      },
      dense = function(model, units = 8) {
        model |> keras3::layer_dense(units = units, activation = "relu")
      },
      output = function(model) {
        model |> keras3::layer_dense(units = 1)
      }
    ),
    mode = "regression"
  )

  tune_spec <- stacks_mlp(dense_units = tune(), fit_epochs = 5L) |>
    parsnip::set_engine("keras")

  # Synthetic data with a clear signal so the candidate models are meaningful
  # and the blend keeps at least one member.
  set.seed(123)
  keras3::set_random_seed(123)

  n <- 150L
  train_dat <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n))
  train_dat$y <- with(
    train_dat,
    3 * x1 - 2 * x2 + 0.5 * x3 + rnorm(n, sd = 0.5)
  )

  rec <- recipes::recipe(y ~ ., data = train_dat) |>
    recipes::step_normalize(recipes::all_numeric_predictors())
  wf <- workflows::workflow(rec, tune_spec)

  params <- tune::extract_parameter_set_dials(wf) |>
    update(dense_units = dials::hidden_units(c(4L, 16L)))
  grid <- dials::grid_regular(params, levels = 2)

  set.seed(456)
  folds <- rsample::vfold_cv(train_dat, v = 2)

  tune_res <- tune::tune_grid(
    wf,
    resamples = folds,
    grid = grid,
    control = stacks::control_stack_grid()
  )

  data_stack <- stacks::stacks() |>
    stacks::add_candidates(tune_res)

  model_stack <- stacks::blend_predictions(
    data_stack,
    penalty = 0.01,
    mixture = 1
  ) |>
    stacks::fit_members()

  preds <- predict(model_stack, new_data = train_dat[1:5, ])

  expect_s3_class(preds, "tbl_df")
  expect_equal(nrow(preds), 5L)
  expect_named(preds, ".pred")
})
