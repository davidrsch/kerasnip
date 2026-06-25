# =============================================================================
# Last-Layer Laplace Approximation interval tests — classification
#
# Verifies that predict(..., type = "conf_int") and
# predict(..., type = "pred_int") work end-to-end with kerasnip
# classification models — binary (2-class softmax) and multi-class softmax
# output layers, sequential and functional Keras APIs.
#
# Note: kerasnip always uses softmax output (units = num_classes) for
# classification, even for binary — the output block receives `num_classes`
# injected by the engine.  There is no separate sigmoid code path.
# =============================================================================

# ---------------------------------------------------------------------------
# Shared sequential layer blocks (classification, softmax)
# ---------------------------------------------------------------------------
make_lla_cls_seq_blocks <- function() {
  input_block <- function(model, input_shape) {
    keras3::keras_model_sequential(input_shape = input_shape)
  }
  dense_block <- function(model, units = 8) {
    model |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(model, num_classes) {
    model |>
      keras3::layer_dense(units = num_classes, activation = "softmax")
  }
  list(input = input_block, dense = dense_block, output = output_block)
}

# ---------------------------------------------------------------------------
# Shared functional layer blocks (classification, softmax)
# ---------------------------------------------------------------------------
make_lla_cls_func_blocks <- function() {
  input_block <- function(input_shape) keras3::layer_input(shape = input_shape)
  dense_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_dense(units = units, activation = "relu")
  }
  output_block <- function(tensor, num_classes) {
    keras3::layer_dense(tensor, units = num_classes, activation = "softmax")
  }
  list(
    main_input = input_block,
    dense = inp_spec(dense_block, "main_input"),
    output = inp_spec(output_block, "dense")
  )
}

# ---------------------------------------------------------------------------
# Helper: assert classification intervals have per-class columns
# ---------------------------------------------------------------------------
expect_valid_class_intervals <- function(result, n_rows, lvl) {
  testthat::expect_s3_class(result, "tbl_df")
  for (cl in lvl) {
    lo_col <- paste0(".pred_lower_", cl)
    hi_col <- paste0(".pred_upper_", cl)
    testthat::expect_true(
      lo_col %in% names(result),
      info = paste("Missing column:", lo_col)
    )
    testthat::expect_true(
      hi_col %in% names(result),
      info = paste("Missing column:", hi_col)
    )
    # Intervals on probability scale [0, 1]
    testthat::expect_true(all(result[[lo_col]] >= 0))
    testthat::expect_true(all(result[[lo_col]] <= 1))
    testthat::expect_true(all(result[[hi_col]] >= 0))
    testthat::expect_true(all(result[[hi_col]] <= 1))
    testthat::expect_true(all(result[[lo_col]] <= result[[hi_col]]))
  }
  testthat::expect_equal(nrow(result), n_rows)
}

# =============================================================================
# Binary Classification (2-class softmax) — Sequential API
# =============================================================================

test_that("LLA: binary sequential conf_int returns per-class columns", {
  skip_if_no_keras()

  model_name <- "lla_bin_ci_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  # Binary iris: setosa vs others
  iris_bin <- iris
  iris_bin$is_setosa <- factor(
    iris_bin$Species == "setosa",
    levels = c(FALSE, TRUE)
  )

  spec <- lla_bin_ci_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    iris_bin
  )
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris_bin)
  result <- predict(fit_obj, iris_bin[1:5, ], type = "conf_int")

  expect_valid_class_intervals(result, 5, c("FALSE", "TRUE"))
})

test_that("LLA: binary sequential pred_int returns valid intervals", {
  skip_if_no_keras()

  model_name <- "lla_bin_pi_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  iris_bin <- iris
  iris_bin$is_setosa <- factor(
    iris_bin$Species == "setosa",
    levels = c(FALSE, TRUE)
  )

  spec <- lla_bin_pi_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    iris_bin
  )
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris_bin)
  result <- predict(fit_obj, iris_bin[1:5, ], type = "pred_int")

  expect_valid_class_intervals(result, 5, c("FALSE", "TRUE"))
})

test_that("LLA: binary pred_int is more extreme than conf_int", {
  skip_if_no_keras()

  model_name <- "lla_bin_ext_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  iris_bin <- iris
  iris_bin$is_setosa <- factor(
    iris_bin$Species == "setosa",
    levels = c(FALSE, TRUE)
  )

  spec <- lla_bin_ext_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    iris_bin
  )
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris_bin)

  ci <- predict(fit_obj, iris_bin[1:5, ], type = "conf_int")
  pi <- predict(fit_obj, iris_bin[1:5, ], type = "pred_int")

  # pred_int lower <= conf_int lower (more extreme / wider)
  testthat::expect_true(
    all(pi$.pred_lower_TRUE <= ci$.pred_lower_TRUE)
  )
  testthat::expect_true(
    all(pi$.pred_upper_TRUE >= ci$.pred_upper_TRUE)
  )
})

test_that("LLA: binary complementary intervals", {
  skip_if_no_keras()

  model_name <- "lla_bin_comp_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  iris_bin <- iris
  iris_bin$is_setosa <- factor(
    iris_bin$Species == "setosa",
    levels = c(FALSE, TRUE)
  )

  spec <- lla_bin_comp_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    iris_bin
  )
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris_bin)
  ci <- predict(fit_obj, iris_bin[1:5, ], type = "conf_int")

  # .pred_lower_FALSE + .pred_upper_TRUE should be approx 1
  complement_sum <- ci$.pred_lower_FALSE + ci$.pred_upper_TRUE
  testthat::expect_true(all(abs(complement_sum - 1) < 0.05))
})

test_that("LLA: binary level argument affects interval width", {
  skip_if_no_keras()

  model_name <- "lla_bin_lvl_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  iris_bin <- iris
  iris_bin$is_setosa <- factor(
    iris_bin$Species == "setosa",
    levels = c(FALSE, TRUE)
  )

  spec <- lla_bin_lvl_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    iris_bin
  )
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris_bin)

  ci80 <- predict(fit_obj, iris_bin[1:5, ], type = "conf_int", level = 0.80)
  ci99 <- predict(fit_obj, iris_bin[1:5, ], type = "conf_int", level = 0.99)

  width80 <- ci80$.pred_upper_TRUE - ci80$.pred_lower_TRUE
  width99 <- ci99$.pred_upper_TRUE - ci99$.pred_lower_TRUE
  testthat::expect_true(all(width80 < width99))
})

# =============================================================================
# Multi-Class Classification (3-class softmax) — Sequential API
# =============================================================================

test_that("LLA: multi-class sequential conf_int returns per-class columns", {
  skip_if_no_keras()

  model_name <- "lla_mc_ci_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  spec <- lla_mc_ci_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(Species ~ ., iris)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris)
  result <- predict(fit_obj, iris[1:5, ], type = "conf_int")

  expect_valid_class_intervals(result, 5, levels(iris$Species))
})

test_that("LLA: multi-class sequential pred_int returns valid intervals", {
  skip_if_no_keras()

  model_name <- "lla_mc_pi_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  spec <- lla_mc_pi_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(Species ~ ., iris)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris)
  result <- predict(fit_obj, iris[1:5, ], type = "pred_int")

  expect_valid_class_intervals(result, 5, levels(iris$Species))
})

test_that("LLA: multi-class level argument respected", {
  skip_if_no_keras()

  model_name <- "lla_mc_lvl_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  spec <- lla_mc_lvl_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(Species ~ ., iris)
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris)

  ci80 <- predict(fit_obj, iris[1:5, ], type = "conf_int", level = 0.80)
  ci99 <- predict(fit_obj, iris[1:5, ], type = "conf_int", level = 0.99)

  for (cl in levels(iris$Species)) {
    lo_col <- paste0(".pred_lower_", cl)
    hi_col <- paste0(".pred_upper_", cl)
    w80 <- ci80[[hi_col]] - ci80[[lo_col]]
    w99 <- ci99[[hi_col]] - ci99[[lo_col]]
    testthat::expect_true(all(w80 < w99))
  }
})

# =============================================================================
# Binary Classification — Functional API
# =============================================================================

test_that("LLA: binary functional conf_int works", {
  skip_if_no_keras()

  model_name <- "lla_bin_ci_func"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_func_blocks(),
    mode = "classification"
  )

  iris_bin <- iris
  iris_bin$is_setosa <- factor(
    iris_bin$Species == "setosa",
    levels = c(FALSE, TRUE)
  )

  spec <- lla_bin_ci_func(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    iris_bin
  )
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris_bin)
  result <- predict(fit_obj, iris_bin[1:5, ], type = "conf_int")

  expect_valid_class_intervals(result, 5, c("FALSE", "TRUE"))
})

# =============================================================================
# saveRDS / readRDS round-trip
# =============================================================================

test_that("LLA: classification intervals survive saveRDS/readRDS", {
  skip_if_no_keras()

  model_name <- "lla_cls_rds_seq"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_sequential_spec(
    model_name = model_name,
    layer_blocks = make_lla_cls_seq_blocks(),
    mode = "classification"
  )

  iris_bin <- iris
  iris_bin$is_setosa <- factor(
    iris_bin$Species == "setosa",
    levels = c(FALSE, TRUE)
  )

  spec <- lla_cls_rds_seq(fit_epochs = 10) |> set_engine("keras")
  rec <- recipe(
    is_setosa ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
    iris_bin
  )
  wf <- workflow(rec, spec)

  set.seed(42)
  fit_obj <- fit(wf, iris_bin)

  ci_before <- predict(fit_obj, iris_bin[1:5, ], type = "conf_int")

  tmp <- tempfile(fileext = ".rds")
  on.exit(unlink(tmp), add = TRUE)
  saveRDS(fit_obj, tmp)
  rm(fit_obj)
  fit_restored <- readRDS(tmp)

  ci_after <- predict(fit_restored, iris_bin[1:5, ], type = "conf_int")

  # Intervals still valid after round-trip
  expect_valid_class_intervals(ci_after, 5, c("FALSE", "TRUE"))
  # Values may shift slightly due to float32 .keras round-trip + MC sampling
  testthat::expect_equal(nrow(ci_before), nrow(ci_after))
  testthat::expect_equal(names(ci_before), names(ci_after))
})
