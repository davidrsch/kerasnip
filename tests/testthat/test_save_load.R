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
  spec <- get(model_name)(fit_epochs = 2, compile_loss = "mse") |>
    parsnip::set_engine("keras")
  wf <- workflows::workflow(
    recipes::recipe(mpg ~ ., data = mtcars),
    spec
  )
  workflows::fit(wf, data = mtcars)
}

fit_func_workflow <- function(model_name) {
  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_func_blocks(),
    mode = "regression"
  )
  spec <- get(model_name)(fit_epochs = 2, compile_loss = "mse") |>
    parsnip::set_engine("keras")
  wf <- workflows::workflow(
    recipes::recipe(mpg ~ ., data = mtcars),
    spec
  )
  workflows::fit(wf, data = mtcars)
}

# ============================================================
# Test 1: Spec carries kerasnip_spec class and metadata
# ============================================================

test_that("spec object has kerasnip_spec class and metadata attributes", {
  skip_if_no_keras()

  model_name <- "sl_class_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_seq_blocks(),
    mode = "regression"
  )

  spec <- get(model_name)(fit_epochs = 2)

  expect_true(inherits(spec, "kerasnip_spec"))
  expect_false(is.null(attr(spec, "kerasnip_layer_blocks")))
  expect_false(is.null(attr(spec, "kerasnip_functional")))
  expect_false(attr(spec, "kerasnip_functional"))
})

# ============================================================
# Test 2: fit() produces kerasnip_model_fit class
# ============================================================

test_that("fit() tags result with kerasnip_model_fit class", {
  skip_if_no_keras()

  model_name <- "sl_class_fit"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  fit_obj <- fit_seq_workflow(model_name)
  mf <- workflows::extract_fit_parsnip(fit_obj)

  expect_true(inherits(mf, "kerasnip_model_fit"))
})

# ============================================================
# Test 3: predict() auto-registers after simulated reload (sequential)
# ============================================================

test_that("predict() auto-registers sequential spec when missing", {
  skip_if_no_keras()

  model_name <- "sl_auto_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  fit_obj <- fit_seq_workflow(model_name)

  rds_path <- tempfile(fileext = ".rds")
  on.exit(unlink(rds_path), add = TRUE)
  saveRDS(fit_obj, rds_path)

  # Simulate new session: remove parsnip registration
  suppressMessages(remove_keras_spec(model_name))
  expect_false(model_exists(model_name))

  loaded <- readRDS(rds_path)

  # predict() should auto-register and succeed
  expect_no_error({
    preds <- predict(loaded, new_data = mtcars[1:3, ])
  })
  expect_equal(nrow(preds), 3)
  expect_true(model_exists(model_name))
})

# ============================================================
# Test 4: predict() auto-registers after simulated reload (functional)
# ============================================================

test_that("predict() auto-registers functional spec when missing", {
  skip_if_no_keras()

  model_name <- "sl_auto_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  fit_obj <- fit_func_workflow(model_name)

  rds_path <- tempfile(fileext = ".rds")
  on.exit(unlink(rds_path), add = TRUE)
  saveRDS(fit_obj, rds_path)

  suppressMessages(remove_keras_spec(model_name))
  expect_false(model_exists(model_name))

  loaded <- readRDS(rds_path)

  expect_no_error({
    preds <- predict(loaded, new_data = mtcars[1:3, ])
  })
  expect_equal(nrow(preds), 3)
  expect_true(model_exists(model_name))
})

# ============================================================
# Test 5: bundle round-trip + predict works (requires bundle package)
# ============================================================

test_that("bundle/unbundle round-trip works and predict succeeds", {
  skip_if_no_keras()
  skip_if_not_installed("bundle")

  model_name <- "sl_bundle_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  fit_obj <- fit_seq_workflow(model_name)

  bundled_path <- tempfile(fileext = ".rds")
  on.exit(unlink(bundled_path), add = TRUE)

  b <- bundle::bundle(fit_obj)
  saveRDS(b, bundled_path)

  suppressMessages(remove_keras_spec(model_name))
  expect_false(model_exists(model_name))

  b2 <- readRDS(bundled_path)
  restored <- bundle::unbundle(b2)

  expect_no_error({
    preds <- predict(restored, new_data = mtcars[1:3, ])
  })
  expect_equal(nrow(preds), 3)
})
