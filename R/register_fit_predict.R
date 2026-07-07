#' Register Fit and Prediction Methods with Parsnip
#'
#' @description
#' This function registers the methods that `parsnip` will use to fit the model
#' and generate predictions.
#'
#' @details
#' This function makes calls to `parsnip::set_fit()` and `parsnip::set_pred()`:
#' - `set_fit()`: Links the model specification to the appropriate generic
#'   fitting engine (`generic_sequential_fit()` or `generic_functional_fit()`).
#'   It also passes the user's `layer_blocks` list as a default argument to
#'   the fitting function.
#' - `set_pred()`: Defines how to generate predictions for different types
#'   ("numeric", "class", "prob"). It specifies the underlying `predict()`
#'   method and the post-processing functions (`keras_postprocess_*`) needed
#'   to format the output into a standardized `tidymodels` tibble.
#'
#' @param model_name The name of the new model.
#' @param mode The model mode ("regression" or "classification").
#' @param layer_blocks The named list of layer block functions, which is passed
#'   as a default argument to the fit function.
#' @param functional A logical. If `TRUE`, registers `generic_functional_fit` as
#'   the fitting engine. Otherwise, registers `generic_sequential_fit`.
#' @return Invisibly returns `NULL`. Called for its side effects.
#' @noRd
register_fit_predict <- function(model_name, mode, layer_blocks, functional) {
  # Fit method
  #
  # `interface = "data.frame"` (rather than "formula") is required so that
  # `fit_xy()` (what `workflows` always calls) routes through
  # parsnip's `xy_xy()` and hands our fit function the outcome data frame
  # exactly as produced by the recipe bake. The "formula" interface instead
  # makes parsnip reconstruct a formula from x/y internally
  # (`.convert_xy_to_form_fit()` + `make_formula()`), which builds the
  # response side as `cbind(col1, col2, ...)` for multi-column outcomes.
  # `cbind()` on two or more factor columns silently discards their factor
  # levels (returns the underlying integer codes as a plain matrix), which
  # breaks `parsnip::check_outcome()` for any multi-output classification
  # spec. Multi-output *regression* happened to work before this change only
  # because `cbind()` of numeric vectors stays numeric.
  parsnip::set_fit(
    model = model_name,
    eng = "keras",
    mode = mode,
    value = list(
      interface = "data.frame",
      protect = c("x", "y"),
      func = c(
        pkg = "kerasnip",
        fun = if (functional) {
          "generic_functional_fit"
        } else {
          "generic_sequential_fit"
        }
      ),
      defaults = list(layer_blocks = layer_blocks)
    )
  )

  # Regression prediction
  if (mode == "regression") {
    # Shared x-processing expression (used by numeric, conf_int, pred_int)
    x_expr <- if (functional) {
      rlang::expr(process_x_functional(new_data)$x_proc)
    } else {
      rlang::expr(process_x_sequential(new_data)$x_proc)
    }

    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "regression",
      type = "numeric",
      value = list(
        pre = NULL,
        post = keras_postprocess_numeric,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = x_expr
        )
      )
    )

    # ---- Laplace confidence intervals ----
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "regression",
      type = "conf_int",
      value = list(
        pre = NULL,
        post = postprocess_intervals_reg,
        func = c(pkg = "kerasnip", fun = "laplace_conf_int_reg"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = x_expr,
          laplace_data = rlang::expr(object$fit$laplace),
          level = rlang::expr(level)
        )
      )
    )

    # ---- Laplace prediction intervals ----
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "regression",
      type = "pred_int",
      value = list(
        pre = NULL,
        post = postprocess_intervals_reg,
        func = c(pkg = "kerasnip", fun = "laplace_pred_int_reg"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = x_expr,
          laplace_data = rlang::expr(object$fit$laplace),
          level = rlang::expr(level)
        )
      )
    )
  } else {
    # Classification predictions
    x_expr <- if (functional) {
      rlang::expr(process_x_functional(new_data)$x_proc)
    } else {
      rlang::expr(process_x_sequential(new_data)$x_proc)
    }

    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "classification",
      type = "class",
      value = list(
        pre = NULL,
        post = keras_postprocess_classes,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = x_expr
        )
      )
    )
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "classification",
      type = "prob",
      value = list(
        pre = NULL,
        post = keras_postprocess_probs,
        func = c(fun = "predict"),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = x_expr
        )
      )
    )

    # ---- Laplace confidence intervals (classification) ----
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "classification",
      type = "conf_int",
      value = list(
        pre = NULL,
        post = postprocess_intervals_cls,
        func = c(
          pkg = "kerasnip",
          fun = "laplace_conf_int_cls"
        ),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = x_expr,
          laplace_data = rlang::expr(object$fit$laplace),
          lvl = rlang::expr(object$fit$lvl),
          level = rlang::expr(level)
        )
      )
    )

    # ---- Laplace prediction intervals (classification) ----
    parsnip::set_pred(
      model = model_name,
      eng = "keras",
      mode = "classification",
      type = "pred_int",
      value = list(
        pre = NULL,
        post = postprocess_intervals_cls,
        func = c(
          pkg = "kerasnip",
          fun = "laplace_pred_int_cls"
        ),
        args = list(
          object = rlang::expr(object$fit$fit),
          x = x_expr,
          laplace_data = rlang::expr(object$fit$laplace),
          lvl = rlang::expr(object$fit$lvl),
          level = rlang::expr(level)
        )
      )
    )
  }
}

##' Post-process Keras Numeric Predictions
#'
#' @description
#' Formats raw numeric predictions from a Keras model into a tibble with the
#' standardized `.pred` column, as required by `tidymodels`.
#'
#' @details
#' For a scalar single-output model, this converts the matrix output from
#' `keras3::predict()` into a single-column tibble. For a single
#' vector-valued output (e.g. multi-step regression, `object$fit$multistep_info`
#' non-`NULL`), it instead builds a nested `.pred` list-column: one inner
#' tibble per row with a `.step` column plus one `.pred`/`.pred_<var>` column
#' per forecasted variable, mirroring the `censored` package's
#' `.pred`/`.eval_time`/`.pred_survival` convention for "several values along
#' one ordered dimension per row", combined with parsnip's `.pred_{outcome}`
#' convention for multiple named variables.
#' @param results A matrix of numeric predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with a `.pred` column.
#' @noRd
keras_postprocess_numeric <- function(results, object) {
  if (is.list(results) && !is.null(names(results))) {
    # Multi-output case: results is a named list of arrays/matrices
    # Combine them into a single tibble with appropriate column names
    combined_preds <- tibble::as_tibble(results)
    # Rename columns to .pred_output_name if there are multiple outputs
    if (length(results) > 1) {
      colnames(combined_preds) <- paste0(".pred_", names(results))
    } else {
      # If only one output, but still a list, name it .pred
      colnames(combined_preds) <- ".pred"
    }
    combined_preds
  } else {
    mat <- as.matrix(results)
    if (ncol(mat) > 1) {
      multistep_pred_column(mat, object$fit$multistep_info)
    } else {
      # Single output case: results is a matrix/array
      tibble::tibble(.pred = as.vector(mat))
    }
  }
}

#' Build a Nested `.pred` Column for Vector-Valued (Multi-Step) Predictions
#'
#' @description
#' Given a `(samples, n_columns)` prediction matrix from a single
#' vector-valued output (one Keras output node, multiple units), builds a
#' `.pred` list-column with one inner tibble per row: a `.step` column plus
#' one prediction column per forecasted variable (`.pred` if there is only
#' one, `.pred_<var>` if there are several).
#'
#' @param mat A numeric matrix, `nrow()` samples by `ncol()` forecasted
#'   values.
#' @param multistep_info A list with `steps` (integer vector, one per
#'   `mat` column) and `vars` (character vector, one per `mat` column), as
#'   produced by `parse_multistep_column_names()`. If `NULL` (no metadata
#'   available), every column is treated as a sequential step of one
#'   unnamed variable.
#' @return A tibble with one `.pred` list-column.
#' @noRd
multistep_pred_column <- function(mat, multistep_info) {
  if (is.null(multistep_info)) {
    multistep_info <- list(
      steps = seq_len(ncol(mat)),
      vars = rep("outcome", ncol(mat))
    )
  }
  steps <- multistep_info$steps
  vars <- multistep_info$vars
  uniq_steps <- sort(unique(steps))
  uniq_vars <- unique(vars)

  pred_list <- purrr::map(seq_len(nrow(mat)), function(i) {
    row_df <- tibble::tibble(.step = uniq_steps)
    for (v in uniq_vars) {
      col_name <- if (length(uniq_vars) > 1) paste0(".pred_", v) else ".pred"
      idx <- multistep_var_col_order(vars, steps, v)
      row_df[[col_name]] <- mat[i, idx]
    }
    row_df
  })

  tibble::tibble(.pred = pred_list)
}

#' Post-process Keras Probability Predictions
#'
#' @description
#' Formats raw probability predictions from a Keras model into a tibble with
#' class-specific column names (e.g., `.pred_class1`, `.pred_class2`).
#'
#' @details
#' This function retrieves the original factor levels from `object$fit$lvl`
#' (which was stored by the fitting engine) and uses them to name the columns.
#' @param results A matrix of probability predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with named columns for each class probability.
#' @noRd
keras_postprocess_probs <- function(results, object) {
  if (is.list(results) && !is.null(names(results))) {
    # Multi-output case: results is a named list of arrays/matrices
    combined_preds <- purrr::map2_dfc(
      results,
      names(results),
      function(res, name) {
        lvls <- object$fit$lvl[[name]]
        # Assuming object$fit$lvl is a named list of levels
        if (is.null(lvls)) {
          # Fallback if levels are not specifically named for this output
          lvls <- paste0("class", seq_len(ncol(res)))
        }
        colnames(res) <- lvls
        tibble::as_tibble(res, .name_repair = "unique") |>
          dplyr::rename_with(~ paste0(".pred_", name, "_", .x))
      }
    )
    combined_preds
  } else {
    # Single output case: results is a matrix/array
    # The levels are now nested inside the fit object
    colnames(results) <- object$fit$lvl
    tibble::as_tibble(results)
  }
}

#' Post-process Keras Class Predictions
#'
#' @description
#' Converts raw probability predictions from a Keras model into a single
#' `.pred_class` column of factor predictions.
#'
#' @details
#' For multiclass models, it finds the class with the highest probability
#' (`which.max`). For binary models, it applies a 0.5 threshold. It uses the
#' levels stored in `object$fit$lvl` to ensure the output factor is correct.
#' @param results A matrix of probability predictions from `predict()`.
#' @param object The `parsnip` model fit object.
#' @return A tibble with a `.pred_class` column containing factor predictions.
#' @noRd
keras_postprocess_classes <- function(results, object) {
  if (is.list(results) && !is.null(names(results))) {
    # Multi-output case: results is a named list of arrays/matrices
    combined_preds <- purrr::map2_dfc(
      results,
      names(results),
      function(res, name) {
        lvls <- object$fit$lvl[[name]]
        # Assuming object$fit$lvl is a named list of levels
        if (is.null(lvls)) {
          # Fallback if levels are not specifically named for this output
          lvls <- paste0("class", seq_len(ncol(res)))
          # This might not be correct for classes, but a placeholder
        }

        if (ncol(res) == 1) {
          # Binary classification
          pred_class <- ifelse(res[, 1] > 0.5, lvls[2], lvls[1])
          pred_class <- factor(pred_class, levels = lvls)
        } else {
          # Multiclass classification
          pred_class_int <- apply(res, 1, which.max)
          pred_class <- lvls[pred_class_int]
          pred_class <- factor(pred_class, levels = lvls)
        }
        tibble::tibble(.pred_class = pred_class) |>
          dplyr::rename_with(~ paste0(".pred_class_", name))
      }
    )
    combined_preds
  } else {
    # Single output case: results is a matrix/array
    # The levels are now nested inside the fit object
    lvls <- object$fit$lvl
    if (ncol(results) == 1) {
      # Binary classification
      pred_class <- ifelse(results[, 1] > 0.5, lvls[2], lvls[1])
      pred_class <- factor(pred_class, levels = lvls)
    } else {
      # Multiclass classification
      pred_class_int <- apply(results, 1, which.max)
      pred_class <- lvls[pred_class_int]
      pred_class <- factor(pred_class, levels = lvls)
    }
    tibble::tibble(.pred_class = pred_class)
  }
}
