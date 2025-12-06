test_that("inp_spec throws error for block with no arguments", {
  block_no_args <- function() {}
  expect_error(
    kerasnip:::inp_spec(block_no_args, "input"),
    "The 'block' function must have at least one argument."
  )
})

test_that("inp_spec throws error for mismatched input_map names", {
  block_with_args <- function(a, b) {}
  input_map_bad <- c(new_a = "a", new_c = "c") # 'c' does not exist
  expect_error(
    kerasnip:::inp_spec(block_with_args, input_map_bad),
    class = "simpleError"
  )
})


test_that("inp_spec supports argument-first mapping", {
  block_with_args <- function(numeric, categorical) {
    list(numeric = numeric, categorical = categorical)
  }
  mapper <- c(
    numeric = "processed_numeric",
    categorical = "processed_categorical"
  )
  wrapped <- kerasnip:::inp_spec(block_with_args, mapper)

  expect_identical(
    names(formals(wrapped))[1:2],
    c("processed_numeric", "processed_categorical")
  )
  res <- wrapped(processed_numeric = 10, processed_categorical = 20)
  expect_identical(res$numeric, 10)
  expect_identical(res$categorical, 20)
})

test_that("inp_spec rejects the legacy input_map orientation", {
  block_with_args <- function(input_a, input_b) {}
  legacy_mapper <- c(processed_a = "input_a", processed_b = "input_b")
  expect_error(
    kerasnip:::inp_spec(block_with_args, legacy_mapper),
    "not found in the block function"
  )
})


test_that("inp_spec throws error for invalid input_map type", {
  block_with_args <- function(a) {}
  expect_error(
    kerasnip:::inp_spec(block_with_args, 123),
    "`input_map` must be a single string or a named character vector."
  )
  expect_error(
    kerasnip:::inp_spec(block_with_args, list(a = "b")),
    "`input_map` must be a single string or a named character vector."
  )
})
