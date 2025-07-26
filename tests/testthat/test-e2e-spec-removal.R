test_that("E2E: Model spec removal works", {
  input_block_rm <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  hidden_block_rm <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_rm <- function(model, num_classes) {
    model |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }

  model_to_remove <- "e2e_mlp_to_remove"

  create_keras_spec(
    model_name = model_to_remove,
    layer_blocks = list(
      input = input_block_rm,
      hidden = hidden_block_rm,
      output = output_block_rm
    ),
    mode = "classification"
  )

  expect_true(exists(model_to_remove, inherits = FALSE))
  expect_true(remove_keras_spec(model_to_remove))
  expect_false(exists(model_to_remove, inherits = FALSE))
  expect_false(remove_keras_spec("a_non_existent_model"))
})
