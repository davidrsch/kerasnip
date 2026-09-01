# Helper to skip tests if Keras is not configured
library(parsnip)
library(recipes)
library(workflows)
library(modeldata)
library(rsample)
library(dials)
library(tune)
library(purrr)
library(dplyr)

skip_if_no_keras <- function() {
  testthat::skip_if_not_installed("keras3")
  testthat::skip_if_not(
    reticulate::py_module_available("keras"),
    "Keras is not available for testing"
  )
}
