#' Compile Keras Models over a Grid of Hyperparameters
#'
#' @description
#' This function allows you to build and compile multiple Keras models based on a
#' `parsnip` model specification and a grid of hyperparameters, without actually
#' fitting them. It's a valuable tool for validating model architectures and
#' catching potential errors early in the modeling process.
#'
#' @details
#' The function operates by iterating through each row of the provided `grid`.
#' For each combination of hyperparameters, it:
#' \enumerate{
#'   \item Constructs the appropriate Keras model (Sequential or Functional) based
#'     on the `spec`.
#'   \item Compiles the model using the specified optimizer, loss, and metrics.
#'   \item Wraps the process in a `try-catch` block to gracefully handle any
#'     errors that might occur during model instantiation or compilation (e.g.,
#'     due to incompatible layer shapes or invalid argument values).
#' }
#' The output is a `tibble` where each row corresponds to a row in the input
#' `grid`. It includes the original hyperparameters, the compiled Keras model
#' object (or a string with the error message if compilation failed), and a
#' summary of the model's architecture.
#'
#' @param spec A `parsnip` model specification created by
#'   `create_keras_sequential_spec()` or `create_keras_functional_spec()`.
#' @param grid A `tibble` or `data.frame` containing the grid of hyperparameters
#'   to evaluate. Each row represents a unique model architecture to be compiled.
#' @param x A data frame or matrix of predictors. This is used to infer the
#'   `input_shape` for the Keras model.
#' @param y A vector of outcomes. This is used to infer the output shape and
#'   the default loss function.
#'
#' @return A `tibble` with the following columns:
#'   \itemize{
#'     \item Columns from the input `grid`.
#'     \item `compiled_model`: A list-column containing the compiled Keras model
#'       objects. If compilation failed for a specific hyperparameter set, this
#'       column will contain a character string with the error message.
#'     \item `model_summary`: A list-column containing a character string with the
#'       output of `keras3::summary_keras_model()` for each successfully compiled
#'       model.
#'   }
#'
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
        summary_char <- utils::capture.output(summary(
          model
        ))
        list(
          compiled_model = list(model),
          model_summary = paste(summary_char, collapse = "\n"),
          error = NA_character_
        )
      },
      error = function(e) {
        list(
          compiled_model = list(NULL),
          model_summary = NA_character_,
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

#' Extract Valid Grid from Compilation Results
#'
#' @description
#' This helper function filters the results from `compile_keras_grid()` to
#' return a new hyperparameter grid containing only the combinations that
#' compiled successfully.
#'
#' @param compiled_grid A tibble, the result of a call to `compile_keras_grid()`.
#'
#' @return A tibble containing the subset of the original grid that resulted in
#'   a successful model compilation (i.e., where the `error` column is `NA`).
#'   The columns for `compiled_model`, `model_summary`, and `error` are removed.
#' @export
extract_valid_grid <- function(compiled_grid) {
  if (
    !is.data.frame(compiled_grid) ||
      !all(
        c("error", "compiled_model", "model_summary") %in% names(compiled_grid)
      )
  ) {
    stop(
      "`compiled_grid` must be a data frame produced by `compile_keras_grid()`."
    )
  }
  
  compiled_grid %>%
    dplyr::filter(is.na(error)) %>%
    dplyr::select(-compiled_model, -model_summary, -error)
}

#' Inform about Compilation Errors
#'
#' @description
#' This helper function inspects the results from `compile_keras_grid()` and
#' prints a formatted summary of any compilation errors that occurred.
#'
#' @param compiled_grid A tibble, the result of a call to `compile_keras_grid()`.
#' @param n The maximum number of errors to display.
#'
#' @return Invisibly returns the input `compiled_grid`. Called for its side
#'   effect of printing to the console.
#' @export
inform_errors <- function(compiled_grid, n = 10) {
  if (!is.data.frame(compiled_grid) || !all(c("error") %in% names(compiled_grid))) {
    stop("`compiled_grid` must be a data frame produced by `compile_keras_grid()`.")
  }
  
  error_grid <- compiled_grid %>%
    dplyr::filter(!is.na(error))
  
  if (nrow(error_grid) > 0) {
    cli::cli_h1("Compilation Errors Summary")
    cli::cli_alert_danger("{nrow(error_grid)} of {nrow(compiled_grid)} models failed to compile.")
    
    for (i in 1:min(nrow(error_grid), n)) {
      row <- error_grid[i, ]
      params <- row %>% dplyr::select(-compiled_model, -model_summary, -error)
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
