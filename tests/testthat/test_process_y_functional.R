test_that("single-column y is processed as a plain vector (regression)", {
  y <- data.frame(target = c(1.5, 2.5, 3.5))
  result <- process_y_functional(y)

  expect_true(is.matrix(result$y_proc))
  expect_false(is.list(result$y_proc) && !is.null(names(result$y_proc)))
  expect_false(result$is_classification)
  expect_null(result$class_levels)
})

test_that("single-column y is processed as one-hot (classification)", {
  y <- data.frame(target = factor(c("a", "b", "a")))
  result <- process_y_functional(y)

  expect_equal(dim(result$y_proc), c(3, 2))
  expect_true(result$is_classification)
  expect_equal(result$num_classes, 2)
})

test_that("multi-column y with layer_blocks = NULL keeps the old behavior", {
  y <- data.frame(output_1 = c(1, 2, 3), output_2 = c(4, 5, 6))
  result <- process_y_functional(y, layer_blocks = NULL)

  expect_true(is.list(result$y_proc))
  expect_equal(names(result$y_proc), c("output_1", "output_2"))
  expect_null(result$multistep_info)
})

test_that("layer_blocks matching column names keeps multi-head behavior", {
  y <- data.frame(output_1 = c(1, 2, 3), output_2 = c(4, 5, 6))
  layer_blocks <- list(
    input = function(input_shape) NULL,
    output_1 = function(tensor) NULL,
    output_2 = function(tensor) NULL
  )
  result <- process_y_functional(y, layer_blocks = layer_blocks)

  expect_true(is.list(result$y_proc))
  expect_equal(names(result$y_proc), c("output_1", "output_2"))
})

test_that("classification y stays multi-head even with one 'output' block", {
  y <- data.frame(
    output_1 = factor(c("a", "b", "a")),
    output_2 = factor(c("x", "y", "x"))
  )
  layer_blocks <- list(
    input = function(input_shape) NULL,
    output = function(tensor) NULL
  )
  result <- process_y_functional(y, layer_blocks = layer_blocks)

  expect_true(is.list(result$y_proc))
  expect_equal(names(result$y_proc), c("output_1", "output_2"))
})

test_that("numeric y with only an 'output' block collapses to one matrix", {
  y <- data.frame(
    lead_1_value = c(1, 2, 3),
    lead_2_value = c(4, 5, 6)
  )
  layer_blocks <- list(
    input = function(input_shape) NULL,
    output = function(tensor, units) NULL
  )
  result <- process_y_functional(y, layer_blocks = layer_blocks)

  expect_true(is.matrix(result$y_proc))
  expect_equal(dim(result$y_proc), c(3, 2))
  expect_false(result$is_classification)
  expect_null(result$class_levels)
  expect_equal(result$multistep_info$steps, c(1, 2))
  expect_equal(result$multistep_info$vars, c("value", "value"))
})

test_that("multistep names fall back to sequential steps when unparseable", {
  y <- data.frame(col_a = c(1, 2, 3), col_b = c(4, 5, 6))
  layer_blocks <- list(
    input = function(input_shape) NULL,
    output = function(tensor, units) NULL
  )
  result <- process_y_functional(y, layer_blocks = layer_blocks)

  expect_equal(result$multistep_info$steps, c(1, 2))
  expect_equal(result$multistep_info$vars, c("outcome", "outcome"))
})

test_that("parse_multistep_column_names parses step_lead() names", {
  col_names <- c("lead_1_temp", "lead_2_temp")
  info <- kerasnip:::parse_multistep_column_names(col_names)
  expect_equal(info$steps, c(1, 2))
  expect_equal(info$vars, c("temp", "temp"))
})

test_that("parse_multistep_column_names parses multiple variables", {
  info <- kerasnip:::parse_multistep_column_names(
    c("lead_1_temp", "lead_1_humidity", "lead_2_temp", "lead_2_humidity")
  )
  expect_equal(info$steps, c(1, 1, 2, 2))
  expect_equal(info$vars, c("temp", "humidity", "temp", "humidity"))
})

test_that("parse_multistep_column_names falls back on unrecognized names", {
  info <- kerasnip:::parse_multistep_column_names(c("foo", "bar", "baz"))
  expect_equal(info$steps, 1:3)
  expect_equal(info$vars, rep("outcome", 3))
})

test_that("parse_multistep_column_names works with a custom prefix", {
  info <- kerasnip:::parse_multistep_column_names(c("h1_temp", "h2_temp"))
  expect_equal(info$steps, c(1, 2))
  expect_equal(info$vars, c("temp", "temp"))
})

test_that("parse_multistep_column_names errors on uneven steps per variable", {
  col_names <- c(
    "lead_1_temp",
    "lead_2_temp",
    "lead_3_temp",
    "lead_1_humidity",
    "lead_2_humidity"
  )
  expect_error(
    kerasnip:::parse_multistep_column_names(col_names),
    "same set of steps"
  )
})
