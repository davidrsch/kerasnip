test_that("E2E: Tuning fit_* and compile_* parameters works", {
  skip_if_no_keras()

  # 1. Define a reusable spec
  model_name <- "tune_fit_compile_spec"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)
  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      dense = function(model, units = 10, input_shape) {
        keras3::keras_model_sequential(input_shape = input_shape) |>
          keras3::layer_dense(units = units)
      },
      output = function(model, num_classes) {
        model |>
          keras3::layer_dense(units = num_classes, activation = "softmax")
      }
    ),
    mode = "classification"
  )

  # 2. Define the model with tunable parameters
  tune_spec <- tune_fit_compile_spec(
    dense_units = tune(),
    fit_batch_size = tune(),
    fit_epochs = tune(),
    compile_optimizer = tune(),
    compile_loss = tune(),
    learn_rate = tune()
  ) |>
    set_engine("keras")

  # 3. Set up workflow and tuning grid
  rec <- recipes::recipe(Species ~ ., data = iris)
  tune_wf <- workflows::workflow(rec, tune_spec)

  params <- tune::extract_parameter_set_dials(tune_wf) |>
    update(
      dense_units = dials::hidden_units(c(8L, 16L)),
      fit_batch_size = dials::batch_size(range = c(16L, 32L), trans = NULL),
      fit_epochs = dials::epochs(range = c(5L, 10L)),
      compile_optimizer = optimizer_function(values = c("adam", "sgd")),
      compile_loss = loss_function_keras(
        values = c("categorical_crossentropy", "kl_divergence")
      ),
      learn_rate = dials::learn_rate(range = c(0.001, 0.01), trans = NULL)
    )

  grid <- dials::grid_regular(params, levels = 2)

  control <- tune::control_grid(save_pred = FALSE, verbose = FALSE)

  # 4. Run tuning
  tune_res <- tune::tune_grid(
    tune_wf,
    resamples = rsample::vfold_cv(iris, v = 2),
    grid = grid,
    control = control
  )

  # 5. Assertions
  expect_s3_class(tune_res, "tune_results")
  metrics <- tune::collect_metrics(tune_res)
  expect_true(all(
    c(
      "dense_units",
      "fit_batch_size",
      "fit_epochs",
      "compile_optimizer",
      "compile_loss",
      "learn_rate"
    ) %in%
      names(metrics)
  ))
  expect_equal(sort(unique(metrics$dense_units)), c(8, 16))
  expect_equal(sort(unique(metrics$fit_batch_size)), c(16, 32))
  expect_equal(sort(unique(metrics$fit_epochs)), c(5, 10))
  expect_equal(sort(unique(metrics$compile_optimizer)), c("adam", "sgd"))
  expect_equal(
    sort(unique(metrics$compile_loss)),
    c("categorical_crossentropy", "kl_divergence")
  )
  expect_equal(sort(unique(metrics$learn_rate)), c(0.001, 0.01))
})
