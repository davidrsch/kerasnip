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

  # A working Keras installation needs both the Python 'keras' module and a
  # backend (e.g., TensorFlow, JAX, or PyTorch). `py_module_available()` is
  # the most reliable, lightweight check; `keras3::is_keras_available()`
  # additionally probes the backend and can be slow, so it is avoided here.
  testthat::skip_if_not(
    reticulate::py_module_available("keras"),
    "Keras and a backend are not available for testing"
  )
}
