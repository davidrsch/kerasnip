test_that("fit arguments can be passed without the fit_ prefix", {
  skip_if_not_installed("keras3")

  create_keras_sequential_spec(
    model_name = "bare_fit_args_spec",
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
  on.exit(
    suppressMessages(remove_keras_spec("bare_fit_args_spec")),
    add = TRUE
  )

  spec <- bare_fit_args_spec(epochs = 3L)

  expect_true("fit_epochs" %in% names(spec$args))
  expect_false("epochs" %in% names(spec$args))
  expect_equal(rlang::get_expr(spec$args$fit_epochs), 3L)
})

test_that("fit_* takes precedence over the bare alias", {
  skip_if_not_installed("keras3")

  create_keras_sequential_spec(
    model_name = "bare_fit_precedence_spec",
    layer_blocks = list(
      input = function(model, input_shape) {
        keras3::keras_model_sequential(input_shape = input_shape)
      },
      output = function(model) {
        model |> keras3::layer_dense(units = 1)
      }
    ),
    mode = "regression"
  )
  on.exit(
    suppressMessages(remove_keras_spec("bare_fit_precedence_spec")),
    add = TRUE
  )

  spec <- bare_fit_precedence_spec(epochs = 3L, fit_epochs = 5L)

  expect_equal(rlang::get_expr(spec$args$fit_epochs), 5L)
})

test_that("bare fit arguments reach the fit engine", {
  skip_if_not_installed("keras3")
  skip_if_not(
    reticulate::py_module_available("keras"),
    "keras not available"
  )

  create_keras_sequential_spec(
    model_name = "bare_fit_args_fit",
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
  on.exit(
    suppressMessages(remove_keras_spec("bare_fit_args_fit")),
    add = TRUE
  )

  spec <- bare_fit_args_fit(epochs = 2L) |> parsnip::set_engine("keras")
  fit_obj <- parsnip::fit(spec, mpg ~ ., data = mtcars)

  expect_equal(length(fit_obj$fit$history$metrics$loss), 2L)
})
