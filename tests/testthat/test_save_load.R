# ============================================================
# Helpers shared across save/load tests
# ============================================================

make_seq_blocks <- function() {
  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block <- function(model, units = 16) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(model) {
    model |> keras3::layer_dense(units = 1)
  }
  list(input = input_block, dense = dense_block, output = output_block)
}

make_func_blocks <- function() {
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  dense_block <- function(tensor, units = 16) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(tensor) keras3::layer_dense(tensor, units = 1)
  list(
    input = input_block,
    dense = inp_spec(dense_block, "input"),
    output = inp_spec(output_block, "dense")
  )
}

fit_seq_workflow <- function(model_name) {
  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_seq_blocks(),
    mode = "regression"
  )
  spec <- get(model_name)(fit_epochs = 2, compile_loss = "mse")
  wf <- workflows::workflow(
    recipes::recipe(mpg ~ ., data = mtcars),
    spec
  )
  fit(wf, data = mtcars)
}

fit_func_workflow <- function(model_name) {
  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_func_blocks(),
    mode = "regression"
  )
  spec <- get(model_name)(fit_epochs = 2, compile_loss = "mse")
  wf <- workflows::workflow(
    recipes::recipe(mpg ~ ., data = mtcars),
    spec
  )
  fit(wf, data = mtcars)
}

# ============================================================
# Test 1: Sequential spec reregistration after simulated reload
# ============================================================

test_that("reregister_keras_spec() restores sequential spec after reload", {
  skip_if_no_keras()

  model_name <- "sl_seq_reg"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  fit_obj <- fit_seq_workflow(model_name)

  rds_path <- tempfile(fileext = ".rds")
  on.exit(unlink(rds_path), add = TRUE)
  saveRDS(fit_obj, rds_path)

  # Simulate a new session by removing the parsnip registration
  suppressMessages(remove_keras_spec(model_name))

  loaded <- readRDS(rds_path)

  # Prediction should fail now — registration is gone
  expect_error(
    predict(loaded, new_data = mtcars[1:3, ]),
    regexp = "not been registered|not registered"
  )

  # Reregister then predict
  reregister_keras_spec(loaded, env = environment())

  preds <- predict(loaded, new_data = mtcars[1:3, ])
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_equal(nrow(preds), 3)
  expect_true(is.numeric(preds$.pred))
})

# ============================================================
# Test 2: Functional spec reregistration after simulated reload
# ============================================================

test_that("reregister_keras_spec() restores functional spec after reload", {
  skip_if_no_keras()

  model_name <- "sl_func_reg"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  fit_obj <- fit_func_workflow(model_name)

  rds_path <- tempfile(fileext = ".rds")
  on.exit(unlink(rds_path), add = TRUE)
  saveRDS(fit_obj, rds_path)

  suppressMessages(remove_keras_spec(model_name))

  loaded <- readRDS(rds_path)

  expect_error(
    predict(loaded, new_data = mtcars[1:3, ]),
    regexp = "not been registered|not registered"
  )

  reregister_keras_spec(loaded, env = environment())

  preds <- predict(loaded, new_data = mtcars[1:3, ])
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_equal(nrow(preds), 3)
  expect_true(is.numeric(preds$.pred))
})

# ============================================================
# Test 3: Full bundle → saveRDS → readRDS → unbundle → reregister round-trip
# ============================================================

test_that("bundle/unbundle + reregister_keras_spec() round-trip works", {
  skip_if_no_keras()
  testthat::skip_if_not_installed("bundle")

  model_name <- "sl_bundle_reg"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  fit_obj <- fit_seq_workflow(model_name)

  rds_path <- tempfile(fileext = ".rds")
  on.exit(unlink(rds_path), add = TRUE)

  bundled <- bundle::bundle(fit_obj)
  saveRDS(bundled, rds_path)

  suppressMessages(remove_keras_spec(model_name))

  bundled_read <- readRDS(rds_path)
  loaded <- bundle::unbundle(bundled_read)

  reregister_keras_spec(loaded, env = environment())

  preds <- predict(loaded, new_data = mtcars[1:3, ])
  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_equal(nrow(preds), 3)
  expect_true(is.numeric(preds$.pred))
})

# ============================================================
# Test 4: Clear error when spec lacks metadata (old-version spec)
# ============================================================

test_that("reregister_keras_spec() gives a clear error for specs without metadata", {
  skip_if_no_keras()

  model_name <- "sl_no_meta"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  fit_obj <- fit_seq_workflow(model_name)
  spec <- workflows::extract_spec_parsnip(fit_obj)

  # Strip the attributes to simulate a spec from an older kerasnip version
  attr(spec, "kerasnip_layer_blocks") <- NULL
  attr(spec, "kerasnip_functional") <- NULL

  expect_error(
    reregister_keras_spec(spec),
    regexp = "kerasnip re-registration metadata"
  )
})
