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
