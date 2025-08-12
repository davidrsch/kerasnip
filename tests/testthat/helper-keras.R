# Helper to skip tests if Keras is not configured
library(parsnip)
library(recipes)
library(workflows)
library(modeldata)
library(rsample)
library(dials)
library(tune)
library(purrr)

skip_if_no_keras <- function() {
  testthat::skip_if_not_installed("keras3")

  # is_keras_available() checks for the python 'keras' module and a backend.
  # This is the most reliable way to check for a working installation.
  # testthat::skip_if_not(
  #   keras3::is_keras_available(),
  #   "Keras 3 and a backend (e.g., tensorflow) are not available for testing"
  # )
}
