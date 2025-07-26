test_that("E2E: Classification spec generation, fitting, and prediction works", {
  skip_if_no_keras()

  input_block_class <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block_class <- function(model, units = 16) {
    model |>
      keras3::layer_dense(units = units, activation = "relu")
  }
  output_block_class <- function(model, num_classes) {
    model |> keras3::layer_dense(units = num_classes, activation = "softmax")
  }

  create_keras_spec(
    model_name = "e2e_mlp_class",
    layer_blocks = list(
      input = input_block_class,
      dense = dense_block_class,
      output = output_block_class
    ),
    mode = "classification"
  )

  spec <- e2e_mlp_class(
    num_dense = 2,
    dense_units = 8,
    epochs = 2
  ) |>
    set_engine("keras")

  # --- Multiclass test ---
  multi_data <- iris
  rec_multi <- recipe(Species ~ ., data = multi_data)
  wf_multi <- workflow(rec_multi, spec)

  expect_no_error(fit_multi <- fit(wf_multi, data = multi_data))
  expect_s3_class(fit_multi, "workflow")

  preds_class_multi <- predict(
    fit_multi,
    new_data = multi_data[1:5, ],
    type = "class"
  )
  expect_s3_class(preds_class_multi, "tbl_df")
  expect_equal(names(preds_class_multi), ".pred_class")
  expect_equal(nrow(preds_class_multi), 5)
  expect_equal(
    levels(preds_class_multi$.pred_class),
    levels(multi_data$Species)
  )

  preds_prob_multi <- predict(
    fit_multi,
    new_data = multi_data[1:5, ],
    type = "prob"
  )
  expect_s3_class(preds_prob_multi, "tbl_df")
  expect_equal(
    names(preds_prob_multi),
    paste0(".pred_", levels(multi_data$Species))
  )
  expect_equal(nrow(preds_prob_multi), 5)
  expect_true(all(abs(rowSums(preds_prob_multi) - 1) < 1e-5))

  # --- Binary test ---
  binary_data <- modeldata::two_class_dat
  rec_bin <- recipe(Class ~ ., data = binary_data)
  wf_bin <- workflow(rec_bin, spec)

  expect_no_error(fit_bin <- fit(wf_bin, data = binary_data))
  expect_s3_class(fit_bin, "workflow")

  preds_class_bin <- predict(
    fit_bin,
    new_data = binary_data[1:5, ],
    type = "class"
  )
  expect_s3_class(preds_class_bin, "tbl_df")
  expect_equal(names(preds_class_bin), ".pred_class")
  expect_equal(nrow(preds_class_bin), 5)
  expect_equal(levels(preds_class_bin$.pred_class), levels(binary_data$Class))

  preds_prob_bin <- predict(
    fit_bin,
    new_data = binary_data[1:5, ],
    type = "prob"
  )
  expect_s3_class(preds_prob_bin, "tbl_df")
  expect_equal(names(preds_prob_bin), c(".pred_Class1", ".pred_Class2"))
  expect_equal(nrow(preds_prob_bin), 5)
  expect_true(all(abs(rowSums(preds_prob_bin) - 1) < 1e-5))
})
