test_that("E2E: Model spec removal works", {
  skip_if_no_keras()

  model_name <- "removable_model"

  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  output_block <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_spec(
    model_name = model_name,
    layer_blocks = list(input = input_block, output = output_block),
    mode = "regression"
  )

  update_method_name <- paste0("update.", model_name)

  expect_true(exists(model_name, inherits = FALSE))
  expect_true(exists(update_method_name, inherits = FALSE))
  expect_error(parsnip:::check_model_doesnt_exist(model_name))

  remove_keras_spec(model_name)

  expect_false(exists(model_name, inherits = FALSE))
  expect_false(exists(update_method_name, inherits = FALSE))
  expect_no_error(parsnip:::check_model_doesnt_exist(model_name))
})
