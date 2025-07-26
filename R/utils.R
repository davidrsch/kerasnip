`%||%` <- function(x, y) {
  # Use y if x is NULL or a parsnip "zapped" argument
  if (is.null(x) || inherits(x, "rlang_zap")) y else x
}

# Environments to store user-registered custom functions
.kerasnip_custom_objects <- new.env(parent = emptyenv())
.kerasnip_custom_objects$optimizers <- list()
.kerasnip_custom_objects$losses <- list()
.kerasnip_custom_objects$metrics <- list()

#' Register a Custom Keras Optimizer
#' @param name The name to register the optimizer under (character).
#' @param optimizer_fn The optimizer function (e.g., a custom function or a partially applied keras optimizer).
#' @export
register_keras_optimizer <- function(name, optimizer_fn) {
  .kerasnip_custom_objects$optimizers[[name]] <- optimizer_fn
  invisible()
}

#' Register a Custom Keras Loss
#' @param name The name to register the loss under (character).
#' @param loss_fn The loss function.
#' @export
register_keras_loss <- function(name, loss_fn) {
  .kerasnip_custom_objects$losses[[name]] <- loss_fn
  invisible()
}

#' Register a Custom Keras Metric
#' @param name The name to register the metric under (character).
#' @param metric_fn The metric function.
#' @export
register_keras_metric <- function(name, metric_fn) {
  .kerasnip_custom_objects$metrics[[name]] <- metric_fn
  invisible()
}

#' Internal helper to retrieve a Keras object by name
#' @noRd
get_keras_object <- function(
  name,
  type = c("optimizer", "loss", "metric"),
  ...
) {
  type <- rlang::arg_match(type)
  storage_name <- paste0(type, "s") # optimizers, losses, metrics

  # 1. Check user-registered custom objects
  custom_fn <- .kerasnip_custom_objects[[storage_name]][[name]]
  if (!is.null(custom_fn)) {
    # If it's an optimizer, pass extra args like learn_rate
    if (type == "optimizer") {
      return(rlang::exec(custom_fn, !!!list(...)))
    }
    return(custom_fn)
  }

  # 2. Check keras3 namespace for a constructor (e.g., optimizer_adam)
  keras_fn_name <- if (type == "optimizer") {
    paste0("optimizer_", name)
  } else {
    paste0(type, "_", name)
  }
  if (exists(keras_fn_name, envir = asNamespace("keras3"))) {
    keras_fn <- get(keras_fn_name, envir = asNamespace("keras3"))
    if (type == "optimizer") {
      return(rlang::exec(keras_fn, !!!list(...)))
    }
    return(keras_fn)
  }

  # 3. If not found, assume it's a string Keras understands directly
  name
}

#' Dials Parameter for Keras Optimizers
#' @param values A character vector of possible optimizers. Defaults to all
#'   known optimizers (keras defaults + custom registered).
#' @keywords internal
#' @export
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
