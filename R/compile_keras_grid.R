#' Compile and Validate Keras Model Architectures
#'
#' @title Compile Keras Models Over a Grid of Hyperparameters
#' @description
#' Pre-compiles Keras models for each hyperparameter combination in a grid.
#'
#' This function is a powerful debugging tool to use before running a full
#' `tune::tune_grid()`. It allows you to quickly validate multiple model
#' architectures, ensuring they can be successfully built and compiled without
#' the time-consuming process of actually fitting them. It helps catch common
#' errors like incompatible layer shapes or invalid argument values early.
#'
#' @details
#' The function iterates through each row of the provided `grid`. For each
#' hyperparameter combination, it attempts to build and compile the Keras model
#' defined by the `spec`. The process is wrapped in a `try-catch` block to
#' gracefully handle and report any errors that occur during model instantiation
#' or compilation.
#'
#' The output is a tibble that mirrors the input `grid`, with additional columns
#' containing the compiled model object or the error message, making it easy to
#' inspect which architectures are valid.
#'
#' @param spec A `parsnip` model specification created by
#'   `create_keras_sequential_spec()` or `create_keras_functional_spec()`.
#' @param grid A `tibble` or `data.frame` containing the grid of hyperparameters
#'   to evaluate. Each row represents a unique model architecture to be compiled.
#' @param x A data frame or matrix of predictors. This is used to infer the
#'   `input_shape` for the Keras model.
#' @param y A vector or factor of outcomes. This is used to infer the output
#'   shape and the default loss function for the Keras model.
#'
#' @return A `tibble` with the following columns:
#'   \itemize{
#'     \item Columns from the input `grid`.
#'     \item `compiled_model`: A list-column containing the compiled Keras model
#'       objects. If compilation failed, the element will be `NULL`.
#'     \item `error`: A list-column containing `NA` for successes or a
#'       character string with the error message for failures.
#'   }
#'
#' @examples
#' \dontrun{
#' if (keras::is_keras_available()) {
#'
#' # 1. Define a kerasnip model specification
#' create_keras_sequential_spec(
#'   model_name = "my_mlp",
#'   layer_blocks = list(
#'     input_block,
#'     hidden_block,
#'     output_block
#'   ),
#'   mode = "classification"
#' )
#'
#' mlp_spec <- my_mlp(
#'   hidden_units = tune(),
#'   compile_loss = "categorical_crossentropy",
#'   compile_optimizer = "adam"
#' )
#'
#' # 2. Create a hyperparameter grid
#' # Include an invalid value (-10) to demonstrate error handling
#' param_grid <- tibble::tibble(
#'   hidden_units = c(32, 64, -10)
#' )
#'
#' # 3. Prepare dummy data
#' x_train <- matrix(rnorm(100 * 10), ncol = 10)
#' y_train <- factor(sample(0:1, 100, replace = TRUE))
#'
#' # 4. Compile models over the grid
#' compiled_grid <- compile_keras_grid(
#'   spec = mlp_spec,
#'   grid = param_grid,
#'   x = x_train,
#'   y = y_train
#' )
#'
#' print(compiled_grid)
#'
#' # 5. Inspect the results
#' # The row with `hidden_units = -10` will show an error.
#' }
#' }
#' @importFrom dplyr bind_rows filter select
#' @importFrom cli cli_h1 cli_alert_danger cli_h2 cli_text cli_bullets cli_code cli_alert_info cli_alert_success
#' @export
compile_keras_grid <- function(spec, grid, x, y) {
  # Input validation
  if (!inherits(spec, "model_spec")) {
    stop("`spec` must be a `parsnip` model specification.")
  }
  if (!is.data.frame(grid)) {
    stop("`grid` must be a data frame or tibble.")
  }

  model_env <- parsnip:::get_model_env()
  model_name <- class(spec)[1]

  fit_info_name <- paste0(model_name, "_fit")
  model_info <- model_env[[fit_info_name]]

  if (is.null(model_info)) {
    stop("Could not find model information for this specification.")
  }

  fit_fun_char <- model_info |>
    purrr::pluck("value") |>
    purrr::pluck(1) |>
    purrr::pluck("func") |>
    purrr::pluck(2)

  build_fn <- if (any(grepl("sequential", fit_fun_char))) {
    build_and_compile_sequential_model
  } else if (any(grepl("functional", fit_fun_char))) {
    build_and_compile_functional_model
  } else {
    stop("Unsupported fit function in model spec.")
  }

  layer_blocks <- model_info |>
    purrr::pluck("value") |>
    purrr::pluck(1) |>
    purrr::pluck("defaults") |>
    purrr::pluck("layer_blocks")
  # Prepare to store results
  results <- purrr::map(1:nrow(grid), function(i) {
    params <- as.list(grid[i, ])
    active_args <- purrr::discard(spec$args, function(arg) {
      inherits(rlang::get_expr(arg), "rlang_zap")
    })

    evaluated_args <- purrr::map(active_args, rlang::eval_tidy)

    args <- list()
    args$x <- x
    args$y <- y
    args$layer_blocks <- layer_blocks

    for (name in names(evaluated_args)) {
      args[[name]] <- evaluated_args[[name]]
    }

    for (name in names(params)) {
      args[[name]] <- params[[name]]
    }

    # Use tryCatch to handle potential errors in model building/compilation
    result <- tryCatch(
      {
        model <- do.call(build_fn, args)
        # Capture the model summary
        list(
          compiled_model = list(model),
          error = NA_character_
        )
      },
      error = function(e) {
        list(
          compiled_model = list(NULL),
          error = as.character(e$message)
        )
      }
    )

    # Combine grid params with the result
    tibble::as_tibble(c(params, result))
  })

  # Combine all results into a single tibble
  dplyr::bind_rows(results)
}

#' Filter a Grid to Only Valid Hyperparameter Sets
#'
#' @title Extract Valid Grid from Compilation Results
#' @description
#' This helper function filters the results from `compile_keras_grid()` to
#' return a new hyperparameter grid containing only the combinations that
#' compiled successfully.
#'
#' @details
#' After running `compile_keras_grid()`, you can use this function to remove
#' problematic hyperparameter combinations before proceeding to the full
#' `tune::tune_grid()`.
#'
#' @param compiled_grid A tibble, the result of a call to `compile_keras_grid()`.
#'
#' @return A tibble containing the subset of the original grid that resulted in
#'   a successful model compilation. The `compiled_model` and `error` columns
#'   are removed, leaving a clean grid ready for tuning.
#'
#' @examples
#' \dontrun{
#' # Continuing the example from `compile_keras_grid`:
#'
#' # `compiled_grid` contains one row with an error.
#' valid_grid <- extract_valid_grid(compiled_grid)
#'
#' # `valid_grid` now only contains the rows that compiled successfully.
#' print(valid_grid)
#'
#' # This clean grid can now be passed to tune::tune_grid().
#' }
#' @export
extract_valid_grid <- function(compiled_grid) {
  if (
    !is.data.frame(compiled_grid) ||
      !all(
        c("error", "compiled_model") %in% names(compiled_grid)
      )
  ) {
    stop(
      "`compiled_grid` must be a data frame produced by `compile_keras_grid()`."
    )
  }
  compiled_grid |>
    dplyr::filter(is.na(error)) |>
    dplyr::select(-c(compiled_model, error))
}

#' Display a Summary of Compilation Errors
#'
#' @title Inform About Compilation Errors
#' @description
#' This helper function inspects the results from `compile_keras_grid()` and
#' prints a formatted, easy-to-read summary of any compilation errors that
#' occurred.
#'
#' @details
#' This is most useful for interactive debugging of complex tuning grids where
#' some hyperparameter combinations may lead to invalid Keras models.
#'
#' @param compiled_grid A tibble, the result of a call to `compile_keras_grid()`.
#' @param n A single integer for the maximum number of distinct errors to
#'   display in detail.
#'
#' @return Invisibly returns the input `compiled_grid`. Called for its side
#'   effect of printing a summary to the console.
#'
#' @examples
#' \dontrun{
#' # Continuing the example from `compile_keras_grid`:
#'
#' # `compiled_grid` contains one row with an error.
#' # This will print a formatted summary of that error.
#' inform_errors(compiled_grid)
#' }
#' @export
inform_errors <- function(compiled_grid, n = 10) {
  if (
    !is.data.frame(compiled_grid) || !all(c("error") %in% names(compiled_grid))
  ) {
    stop(
      "`compiled_grid` must be a data frame produced by `compile_keras_grid()`."
    )
  }
  error_grid <- compiled_grid |>
    dplyr::filter(!is.na(error))
  if (nrow(error_grid) > 0) {
    cli::cli_h1("Compilation Errors Summary")
    cli::cli_alert_danger(
      "{nrow(error_grid)} of {nrow(compiled_grid)} models failed to compile."
    )

    for (i in 1:min(nrow(error_grid), n)) {
      row <- error_grid[i, ]
      params <- row |> dplyr::select(-c(compiled_model, error))
      cli::cli_h2("Error {i}/{nrow(error_grid)}")
      cli::cli_text("Hyperparameters:")
      cli::cli_bullets(paste0(names(params), ": ", as.character(params)))
      cli::cli_text("Error Message:")
      cli::cli_code(row$error)
    }
    if (nrow(error_grid) > n) {
      cli::cli_alert_info("... and {nrow(error_grid) - n} more errors.")
    }
  } else {
    cli::cli_alert_success("All models compiled successfully!")
  }
  invisible(compiled_grid)
}
