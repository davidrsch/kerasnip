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
#'
#' @description
#' Allows users to register a custom optimizer function so it can be used by
#' name within `kerasnip` model specifications and tuned with `dials`.
#'
#' @details
#' Registered optimizers are stored in an internal environment. When a model is
#' compiled, `kerasnip` will first check this internal registry for an optimizer
#' matching the provided name before checking the `keras3` package.
#'
#' The `optimizer_fn` can be a simple function or a partially applied function
#' using `purrr::partial()`. This is useful for creating versions of Keras
#' optimizers with specific settings.
#'
#' @param name The name to register the optimizer under (character).
#' @param optimizer_fn The optimizer function. It should return a Keras
#'   optimizer object.
#' @seealso [register_keras_loss()], [register_keras_metric()]
#' @export
#' @examples
#' if (requireNamespace("keras3", quietly = TRUE)) {
#'   # Register a custom version of Adam with a different default beta_1
#'   my_adam <- purrr::partial(keras3::optimizer_adam, beta_1 = 0.8)
#'   register_keras_optimizer("my_adam", my_adam)
#'
#'   # Now "my_adam" can be used as a string in a model spec, e.g.,
#'   # my_model_spec(compile_optimizer = "my_adam")
#' }
register_keras_optimizer <- function(name, optimizer_fn) {
  .kerasnip_custom_objects$optimizers[[name]] <- optimizer_fn
  invisible()
}

#' Register a Custom Keras Loss
#'
#' @description
#' Allows users to register a custom loss function so it can be used by name
#' within `kerasnip` model specifications and tuned with `dials`.
#'
#' @details
#' Registered losses are stored in an internal environment. When a model is
#' compiled, `kerasnip` will first check this internal registry for a loss
#' matching the provided name before checking the `keras3` package.
#'
#' @param name The name to register the loss under (character).
#' @param loss_fn The loss function.
#' @seealso [register_keras_optimizer()], [register_keras_metric()]
#' @export
register_keras_loss <- function(name, loss_fn) {
  .kerasnip_custom_objects$losses[[name]] <- loss_fn
  invisible()
}

#' Register a Custom Keras Metric
#'
#' @description
#' Allows users to register a custom metric function so it can be used by name
#' within `kerasnip` model specifications.
#'
#' @details
#' Registered metrics are stored in an internal environment. When a model is
#' compiled, `kerasnip` will first check this internal registry for a metric
#' matching the provided name before checking the `keras3` package.
#'
#' @param name The name to register the metric under (character).
#' @param metric_fn The metric function.
#' @seealso [register_keras_optimizer()], [register_keras_loss()]
#' @export
register_keras_metric <- function(name, metric_fn) {
  .kerasnip_custom_objects$metrics[[name]] <- metric_fn
  invisible()
}

#' Internal helper to retrieve a Keras object by name from the registry
#'
#' @description
#' Resolves a string name into a Keras object (optimizer, loss, or metric)
#' by searching in a specific order.
#'
#' @details
#' The lookup order is:
#' 1.  User-registered custom objects via `register_keras_*()`.
#' 2.  Standard Keras constructors in the `keras3` package (e.g., `optimizer_adam`).
#' 3.  If not found, the original string is returned, assuming Keras can handle it.
#'
#' For optimizers, it also passes along any `...` arguments (like `learning_rate`)
#' to the constructor function.
#'
#' @param name The string name of the object.
#' @param type The type of object ("optimizer", "loss", or "metric").
#' @param ... Additional arguments passed to the optimizer constructor.
#' @return A Keras object or a string name.
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

#' Process Predictor Input for Keras
#'
#' @description
#' Preprocesses predictor data (`x`) into a format suitable for Keras models.
#' Handles both tabular data and list-columns of arrays (e.g., for images).
#'
#' @param x A data frame or matrix of predictors.
#' @return A list containing:
#'   - `x_proc`: The processed predictor data (matrix or array).
#'   - `input_shape`: The determined input shape for the Keras model.
#' @keywords internal
#' @export
process_x <- function(x) {
  if (is.data.frame(x) && ncol(x) == 1 && is.list(x[[1]])) {
    # Assumes a single predictor column containing a list of arrays.
    # We stack them into a single higher-dimensional array.
    x_proc <- do.call(abind::abind, c(x[[1]], list(along = 0)))
  } else {
    x_proc <- as.matrix(x)
  }
  input_shape <- if (length(dim(x_proc)) > 2) dim(x_proc)[-1] else ncol(x_proc)
  list(x_proc = x_proc, input_shape = input_shape)
}

#' Process Outcome Input for Keras
#'
#' @description
#' Preprocesses outcome data (`y`) into a format suitable for Keras models.
#' Handles both regression (numeric) and classification (factor) outcomes,
#' including one-hot encoding for classification.
#'
#' @param y A vector of outcomes.
#' @param is_classification Logical, optional. If `TRUE`, treats `y` as
#'   classification. If `FALSE`, treats as regression. If `NULL` (default),
#'   it's determined from `is.factor(y)`.
#' @param class_levels Character vector, optional. The factor levels for
#'   classification outcomes. If `NULL` (default), determined from `levels(y)`.
#' @return A list containing:
#'   - `y_proc`: The processed outcome data (matrix or one-hot encoded array).
#'   - `is_classification`: Logical, indicating if `y` was treated as classification.
#'   - `num_classes`: Integer, the number of classes for classification, or `NULL`.
#'   - `class_levels`: Character vector, the factor levels for classification, or `NULL`.
#' @importFrom keras3 to_categorical
#' @keywords internal
#' @export
process_y <- function(y, is_classification = NULL, class_levels = NULL) {
  # If y is a data frame/tibble, extract the first column
  if (is.data.frame(y)) {
    y <- y[[1]]
  }

  if (is.null(is_classification)) {
    is_classification <- is.factor(y)
  }

  y_proc <- NULL
  num_classes <- NULL
  if (is_classification) {
    if (is.null(class_levels)) {
      class_levels <- levels(y)
    }
    num_classes <- length(class_levels)
    y_factored <- factor(y, levels = class_levels)
    y_proc <- keras3::to_categorical(
      as.numeric(y_factored) - 1,
      num_classes = num_classes
    )
  } else {
    y_proc <- as.matrix(y)
  }
  list(
    y_proc = y_proc,
    is_classification = is_classification,
    num_classes = num_classes,
    class_levels = class_levels
  )
}
