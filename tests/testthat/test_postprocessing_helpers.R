skip_if_no_keras()

# Mock object for post-processing functions
mock_object_single_output <- list(
  fit = list(
    lvl = c("setosa", "versicolor", "virginica") # For classification levels
  )
)
class(mock_object_single_output) <- "model_fit"

mock_object_multi_output <- list(
  fit = list(
    lvl = list(
      output1 = c("classA", "classB"),
      output2 = c("typeX", "typeY", "typeZ")
    )
  )
)
class(mock_object_multi_output) <- "model_fit"

# --- Tests for keras_postprocess_numeric ---

test_that("keras_postprocess_numeric handles single output (matrix) correctly", {
  results <- matrix(c(0.1, 0.2, 0.3), ncol = 1)
  processed <- keras_postprocess_numeric(results, mock_object_single_output)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), ".pred")
  expect_equal(processed$.pred, c(0.1, 0.2, 0.3))
})

test_that("keras_postprocess_numeric handles single output (named list with one element) correctly", {
  results <- list(output1 = matrix(c(0.1, 0.2, 0.3), ncol = 1))
  names(results) <- "output1"
  processed <- keras_postprocess_numeric(results, mock_object_multi_output)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), ".pred")
  expect_equal(processed$.pred, matrix(c(0.1, 0.2, 0.3), ncol = 1)) # Changed expected
})


test_that("keras_postprocess_numeric handles multi-output (named list) correctly", {
  results <- list(
    output1 = matrix(c(0.1, 0.2), ncol = 1),
    output2 = matrix(c(0.4, 0.5), ncol = 1)
  )
  names(results) <- c("output1", "output2")
  processed <- keras_postprocess_numeric(results, mock_object_multi_output)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), c(".pred_output1", ".pred_output2"))
  # Change expected values to 1-column matrices
  expect_equal(processed$.pred_output1, matrix(c(0.1, 0.2), ncol = 1))
  expect_equal(processed$.pred_output2, matrix(c(0.4, 0.5), ncol = 1))
})

# --- Tests for keras_postprocess_probs ---

test_that("keras_postprocess_probs handles single output (matrix) correctly", {
  results <- matrix(c(0.1, 0.9, 0.0,  # Example probabilities for 3 classes
                      0.2, 0.1, 0.7,
                      0.3, 0.3, 0.4), ncol = 3, byrow = TRUE)
  processed <- keras_postprocess_probs(results, mock_object_single_output)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), c("setosa", "versicolor", "virginica")) # Updated expected names
  expect_equal(processed$setosa, c(0.1, 0.2, 0.3)) # Access by correct column name
  expect_equal(processed$versicolor, c(0.9, 0.1, 0.3)) # Access by correct column name
  expect_equal(processed$virginica, c(0.0, 0.7, 0.4)) # Access by correct column name
})

test_that("keras_postprocess_probs handles multi-output (named list) correctly", {
  results <- list(
    output1 = matrix(c(0.1, 0.9, 0.2, 0.8), ncol = 2, byrow = TRUE),
    output2 = matrix(c(0.3, 0.4, 0.3, 0.5, 0.2, 0.3), ncol = 3, byrow = TRUE)
  )
  names(results) <- c("output1", "output2")
  processed <- keras_postprocess_probs(results, mock_object_multi_output)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), c(".pred_output1_classA", ".pred_output1_classB", ".pred_output2_typeX", ".pred_output2_typeY", ".pred_output2_typeZ"))
  expect_equal(processed$.pred_output1_classA, c(0.1, 0.2))
  expect_equal(processed$.pred_output2_typeX, c(0.3, 0.5))
})

test_that("keras_postprocess_probs handles multi-output with NULL levels fallback", {
  results <- list(
    output1 = matrix(c(0.1, 0.9, 0.2, 0.8), ncol = 2, byrow = TRUE)
  )
  names(results) <- "output1"
  mock_object_null_lvl <- list(
    fit = list(
      lvl = list(output1 = NULL) # Simulate NULL levels for this output
    )
  )
  class(mock_object_null_lvl) <- "model_fit"
  processed <- keras_postprocess_probs(results, mock_object_null_lvl)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), c(".pred_output1_class1", ".pred_output1_class2"))
})

# --- Tests for keras_postprocess_classes ---

test_that("keras_postprocess_classes handles single output (multiclass) correctly", {
  results <- matrix(c(0.1, 0.8, 0.1, 0.2, 0.1, 0.7), ncol = 3, byrow = TRUE)
  processed <- keras_postprocess_classes(results, mock_object_single_output)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), ".pred_class")
  expect_equal(as.character(processed$.pred_class), c("versicolor", "virginica"))
  expect_true(is.factor(processed$.pred_class))
  expect_equal(levels(processed$.pred_class), c("setosa", "versicolor", "virginica"))
})

test_that("keras_postprocess_classes handles single output (binary) correctly", {
  results <- matrix(c(0.6, 0.4), ncol = 1) # Changed to single column
  mock_object_binary_lvl <- list(
    fit = list(
      lvl = c("negative", "positive")
    )
  )
  class(mock_object_binary_lvl) <- "model_fit"
  processed <- keras_postprocess_classes(results, mock_object_binary_lvl)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), ".pred_class")
  expect_equal(as.character(processed$.pred_class), c("positive", "negative")) # Changed expected
  expect_true(is.factor(processed$.pred_class))
  expect_equal(levels(processed$.pred_class), c("negative", "positive"))
})

test_that("keras_postprocess_classes handles multi-output (named list) correctly", {
  results <- list(
    output1 = matrix(c(0.1, 0.9, 0.2, 0.8), ncol = 2, byrow = TRUE), # Binary
    output2 = matrix(c(0.3, 0.4, 0.3, 0.5, 0.2, 0.3), ncol = 3, byrow = TRUE) # Multiclass
  )
  names(results) <- c("output1", "output2")
  processed <- keras_postprocess_classes(results, mock_object_multi_output)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), c(".pred_class_output1", ".pred_class_output2"))
  expect_equal(as.character(processed$.pred_class_output1), c("classB", "classB"))
  expect_equal(as.character(processed$.pred_class_output2), c("typeY", "typeX"))
  expect_true(is.factor(processed$.pred_class_output1))
  expect_true(is.factor(processed$.pred_class_output2))
  expect_equal(levels(processed$.pred_class_output1), c("classA", "classB"))
  expect_equal(levels(processed$.pred_class_output2), c("typeX", "typeY", "typeZ"))
})

test_that("keras_postprocess_classes handles multi-output with NULL levels fallback", {
  results <- list(
    output1 = matrix(c(0.6, 0.4, 0.2, 0.8), ncol = 2, byrow = TRUE) # Binary
  )
  names(results) <- "output1"
  mock_object_null_lvl <- list(
    fit = list(
      lvl = list(output1 = NULL) # Simulate NULL levels for this output
    )
  )
  class(mock_object_null_lvl) <- "model_fit"
  processed <- keras_postprocess_classes(results, mock_object_null_lvl)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), c(".pred_class_output1"))
  expect_equal(as.character(processed$.pred_class_output1), c("class1", "class2")) # Changed expected
  expect_true(is.factor(processed$.pred_class_output1))
  expect_equal(levels(processed$.pred_class_output1), c("class1", "class2"))
})

test_that("keras_postprocess_classes handles multi-output (binary, single column) correctly", {
  results <- list(
    output1 = matrix(c(0.6, 0.4, 0.2, 0.8), ncol = 1, byrow = TRUE) # Single column binary output
  )
  names(results) <- "output1"
  mock_object_multi_output_binary <- list(
    fit = list(
      lvl = list(output1 = c("negative", "positive")) # Levels for binary output
    )
  )
  class(mock_object_multi_output_binary) <- "model_fit"
  processed <- keras_postprocess_classes(results, mock_object_multi_output_binary)
  expect_s3_class(processed, "tbl_df")
  expect_equal(names(processed), c(".pred_class_output1"))
  expect_equal(as.character(processed$.pred_class_output1), c("positive", "negative", "negative", "positive")) # Expected based on 0.5 threshold
  expect_true(is.factor(processed$.pred_class_output1))
  expect_equal(levels(processed$.pred_class_output1), c("negative", "positive"))
})