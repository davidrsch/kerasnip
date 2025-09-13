test_that("autoplot works with multiple hidden units parameters", {
  skip_if_no_keras()
  skip_if_not_installed("ggplot2")

  # 1. Define a spec with multiple hidden unit parameters
  model_name <- "autoplot_spec"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)
  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = list(
      input = function(model, input_shape) {
        keras3::keras_model_sequential(input_shape = input_shape)
      },
      dense1 = function(model, units = 10) {
        model |> keras3::layer_dense(units = units)
      },
      dense2 = function(model, units = 10) {
        model |> keras3::layer_dense(units = units)
      },
      output = function(model, num_classes) {
        model |>
          keras3::layer_dense(units = num_classes, activation = "softmax")
      }
    ),
    mode = "classification"
  )

  tune_spec <- autoplot_spec(
    dense1_units = tune(id = "denseone"),
    dense2_units = tune(id = "densetwo")
  ) |>
    set_engine("keras")

  # 2. Set up workflow and tuning grid
  rec <- recipes::recipe(Species ~ ., data = iris)
  tune_wf <- workflows::workflow(rec, tune_spec)

  params <- tune::extract_parameter_set_dials(tune_wf)

  # The user code should not need to change.
  # `hidden_units` will be `kerasnip::hidden_units` which auto-detects the id.
  params <- params |>
    update(
      denseone = hidden_units(range = c(4L, 8L)),
      densetwo = hidden_units(range = c(4L, 8L))
    )
  params$name
  params$id
  params$source
  params$component
  params$component_id
  params$object

  grid <- dials::grid_regular(params, levels = 2)
  control <- tune::control_grid(save_pred = FALSE, verbose = FALSE)

  # 3. Run tuning
  tune_res <- tune::tune_grid(
    tune_wf,
    resamples = rsample::vfold_cv(iris, v = 2),
    grid = grid,
    control = control
  )

  # 4. Assert that autoplot works without error
  expect_no_error(
    ggplot2::autoplot(tune_res)
  )
})
