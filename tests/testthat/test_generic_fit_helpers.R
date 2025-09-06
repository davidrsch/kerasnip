# Mock get_keras_object to isolate the logic of collect_compile_args
mock_get_keras_object <- function(name, type, ...) {
  # Return a simple string representation for testing purposes
  paste0("mocked_", type, "_", name)
}

# Mock optimizer to avoid keras dependency
mock_optimizer_adam <- function(...) {
  "mocked_optimizer_adam"
}

test_that("collect_compile_args handles single-output cases correctly", {
  # Mock the keras3::optimizer_adam function
  testthat::with_mocked_bindings(
    .env = as.environment("package:kerasnip"),
    get_keras_object = mock_get_keras_object,
    {
      # Case 1: Single output, non-character loss and metrics
      dummy_loss_obj <- structure(list(), class = "dummy_loss")
      dummy_metric_obj <- structure(list(), class = "dummy_metric")

      args <- collect_compile_args(
        all_args = list(
          compile_loss = dummy_loss_obj,
          compile_metrics = list(dummy_metric_obj)
        ),
        learn_rate = 0.01,
        default_loss = "mse",
        default_metrics = "mae"
      )
      expect_equal(args$loss, dummy_loss_obj)
      expect_equal(args$metrics, list(dummy_metric_obj))
    }
  )
})

test_that("collect_compile_args handles multi-output cases correctly", {
  testthat::with_mocked_bindings(
    .env = as.environment("package:kerasnip"),
    get_keras_object = mock_get_keras_object,
    {
      # Case 2: Multi-output, single string for loss and metrics
      args <- collect_compile_args(
        all_args = list(
          compile_loss = "categorical_crossentropy",
          compile_metrics = "accuracy"
        ),
        learn_rate = 0.01,
        default_loss = list(out1 = "mse", out2 = "mae"),
        default_metrics = list(out1 = "mse", out2 = "mae")
      )
      expect_equal(args$loss, "mocked_loss_categorical_crossentropy")
      expect_equal(args$metrics, "mocked_metric_accuracy")

      # Case 3: Multi-output, named list with mixed types
      dummy_loss_obj_2 <- structure(list(), class = "dummy_loss_2")
      args_mixed <- collect_compile_args(
        all_args = list(
          compile_loss = list(out1 = "mae", out2 = dummy_loss_obj_2)
        ),
        learn_rate = 0.01,
        default_loss = list(out1 = "mse", out2 = "mae"),
        default_metrics = list(out1 = "mse", out2 = "mae")
      )
      expect_equal(args_mixed$loss$out1, "mocked_loss_mae")
      expect_equal(args_mixed$loss$out2, dummy_loss_obj_2)
    }
  )
})

test_that("collect_compile_args handles named list of metrics (multi-output) correctly", {
  testthat::with_mocked_bindings(
    .env = as.environment("package:kerasnip"),
    get_keras_object = mock_get_keras_object,
    {
      # Test case: Named list of metrics with mixed types (character and object)
      dummy_metric_obj_3 <- structure(list(), class = "dummy_metric_3")
      args_mixed_metrics <- collect_compile_args(
        all_args = list(
          compile_metrics = list(out1 = "accuracy", out2 = dummy_metric_obj_3)
        ),
        learn_rate = 0.01,
        default_loss = list(out1 = "mse", out2 = "mae"),
        default_metrics = list(out1 = "mse", out2 = "mae") # Important: default_metrics must be a named list for this path
      )
      expect_equal(args_mixed_metrics$metrics$out1, "mocked_metric_accuracy")
      expect_equal(args_mixed_metrics$metrics$out2, dummy_metric_obj_3)

      # Test case: Named list of metrics with all characters
      args_all_char_metrics <- collect_compile_args(
        all_args = list(
          compile_metrics = list(out1 = "accuracy", out2 = "mse")
        ),
        learn_rate = 0.01,
        default_loss = list(out1 = "mse", out2 = "mae"),
        default_metrics = list(out1 = "mse", out2 = "mae")
      )
      expect_equal(args_all_char_metrics$metrics$out1, "mocked_metric_accuracy")
      expect_equal(args_all_char_metrics$metrics$out2, "mocked_metric_mse")

      # Test case: Named list of metrics with all objects
      dummy_metric_obj_4 <- structure(list(), class = "dummy_metric_4")
      dummy_metric_obj_5 <- structure(list(), class = "dummy_metric_5")
      args_all_obj_metrics <- collect_compile_args(
        all_args = list(
          compile_metrics = list(
            out1 = dummy_metric_obj_4,
            out2 = dummy_metric_obj_5
          )
        ),
        learn_rate = 0.01,
        default_loss = list(out1 = "mse", out2 = "mae"),
        default_metrics = list(out1 = "mse", out2 = "mae")
      )
      expect_equal(args_all_obj_metrics$metrics$out1, dummy_metric_obj_4)
      expect_equal(args_all_obj_metrics$metrics$out2, dummy_metric_obj_5)
    }
  )
})

test_that("collect_compile_args throws errors for invalid multi-output args", {
  # Case 4: Multi-output, invalid loss argument
  expect_error(
    collect_compile_args(
      all_args = list(compile_loss = list("a", "b")), # Unnamed list
      learn_rate = 0.01,
      default_loss = list(out1 = "mse", out2 = "mae"),
      default_metrics = list(out1 = "mse", out2 = "mae")
    ),
    "For multiple outputs, 'compile_loss' must be a single string or a named list of losses."
  )

  # Case 5: Multi-output, invalid metrics argument
  expect_error(
    collect_compile_args(
      all_args = list(compile_metrics = list("a", "b")), # Unnamed list
      learn_rate = 0.01,
      default_loss = list(out1 = "mse", out2 = "mae"),
      default_metrics = list(out1 = "mse", out2 = "mae")
    ),
    "For multiple outputs, 'compile_metrics' must be a single string or a named list of metrics."
  )
})
