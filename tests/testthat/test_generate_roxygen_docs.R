test_that("generate_roxygen_docs handles fallback case for block params", {
  # This test covers the case where a parameter is passed in `all_args`
  # that is not a num_, fit_, or compile_ param, and does not match
  # any of the layer_blocks. This is a fallback case that should not
  # happen in normal operation but is tested for completeness.
  doc_string <- kerasnip:::generate_roxygen_docs(
    model_name = "test_model",
    layer_blocks = list(
      dense = function(units = 10) {}
    ),
    all_args = list(
      unmatched_param = 1
    ),
    functional = FALSE
  )
  expect_true(grepl("@param unmatched_param A model parameter.", doc_string))
})
