test_that("keras_evaluate throws error for missing processing functions", {
  # Create a dummy fit object that is missing the processing functions
  dummy_fit_object <- list(
    fit = list(
      fit = "dummy_keras_model"
      # process_x and process_y are missing
    )
  )
  class(dummy_fit_object) <- "model_fit"

  expect_error(
    keras_evaluate(dummy_fit_object, x = mtcars[, -1], y = mtcars$mpg),
    "Could not find processing functions in the model fit object."
  )
})
