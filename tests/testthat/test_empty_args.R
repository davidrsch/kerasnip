test_that("collect_spec_args handles empty keras arg names", {
  testthat::with_mocked_bindings(
    .env = as.environment("package:kerasnip"),
    keras_fit_arg_names = character(0),
    keras_compile_arg_names = character(0),
    {
      args_info <- kerasnip:::collect_spec_args(
        layer_blocks = list(dense = function(model, units = 10) {}),
        functional = FALSE
      )
      # Expect only num_dense, dense_units, and learn_rate
      expected_args <- c("num_dense", "dense_units", "learn_rate")
      expect_equal(sort(names(args_info$all_args)), sort(expected_args))
      expect_equal(sort(args_info$parsnip_names), sort(expected_args))
    }
  )
})
