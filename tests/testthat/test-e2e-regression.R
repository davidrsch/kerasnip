test_that("E2E: Regression spec generation, fitting, and prediction works", {
  skip_if_no_keras()

  input_block_reg <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_reg <- function(model, units = 16, dropout = 0.1) {
    model |>
      keras3::layer_dense(units = units, activation = "relu") |>
      keras3::layer_dropout(rate = dropout)
  }
  output_block_reg <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }

  create_keras_sequential_spec(
    model_name = "e2e_mlp_reg",
    layer_blocks = list(
      input = input_block_reg,
      dense = dense_block_reg,
      output = output_block_reg
    ),
    mode = "regression"
  )

  spec <- e2e_mlp_reg(
    num_dense = 2,
    dense_units = 8,
    epochs = 2,
    learn_rate = 0.01
  ) |>
    set_engine("keras")

  data <- mtcars
  rec <- recipe(mpg ~ ., data = data)
  wf <- workflow(rec, spec)

  expect_no_error(
    fit_obj <- fit(wf, data = data)
  )
  expect_s3_class(fit_obj, "workflow")

  preds <- predict(fit_obj, new_data = data[1:5, ])
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_equal(nrow(preds), 5)
  expect_true(is.numeric(preds$.pred))
})
