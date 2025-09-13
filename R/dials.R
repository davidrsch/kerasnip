#' Dials Parameter for Keras Optimizers
#' @param values A character vector of possible optimizers. Defaults to all
#'   known optimizers (keras defaults + custom registered).
#' @keywords internal
#' @export
#' @return A `dials` parameter object for Keras optimizers.
optimizer_function <- function(values = NULL) {
  if (is.null(values)) {
    values <- unique(c(
      keras_optimizers,
      names(.kerasnip_custom_objects$optimizers)
    ))
  }
  dials::new_qual_param(
    type = "character",
    values = values,
    label = c(optimizer_function = "Optimizer Function"),
    finalize = NULL
  )
}

#' Dials Parameter for Keras Loss Functions
#' @param values A character vector of possible loss functions. Defaults to all
#'   known losses (keras defaults + custom registered).
#' @keywords internal
#' @export
#' @return A `dials` parameter object for Keras loss.
loss_function_keras <- function(values = NULL) {
  if (is.null(values)) {
    values <- unique(c(keras_losses, names(.kerasnip_custom_objects$losses)))
  }
  dials::new_qual_param(
    type = "character",
    values = values,
    label = c(loss_function_keras = "Loss Function"),
    finalize = NULL
  )
}
