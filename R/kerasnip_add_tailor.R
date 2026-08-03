# Shared helpers for adapting a single output's/step's predictions to/from
# the standard `.pred` / `.pred_class` / `.pred_<level>` shape `tailor`/
# `probably` expect, and back to kerasnip's own shape (`.pred_<output>`
# columns for multi-output; the nested `.pred` list-column for multistep).
# Used by both `kerasnip_add_tailor()` and (indirectly) by manual
# `kerasnip_output_view()`/`kerasnip_step_view()` usage.

#' Standard-Shape Predictions from an Output or Step View
#'
#' @description
#' Predicts from `view` in the shape `kerasnip_add_tailor()` needs to fit or
#' apply a `tailor`: `.pred` for a step view or a regression output view, or
#' `.pred_class` + `.pred_<level>` columns for a classification output view.
#'
#' @param view A `kerasnip_output_view` or `kerasnip_step_view`.
#' @param data A data frame of predictors.
#' @return A tibble in the standard single-output prediction shape.
#' @keywords internal
#' @noRd
kerasnip_view_predictions <- function(view, data) {
  if (inherits(view, "kerasnip_step_view")) {
    return(predict(view, new_data = data, type = "numeric"))
  }
  if (view$mode == "classification") {
    preds_class <- predict(view, new_data = data, type = "class")
    preds_prob <- predict(view, new_data = data, type = "prob")
    dplyr::bind_cols(preds_class, preds_prob)
  } else {
    predict(view, new_data = data, type = "numeric")
  }
}

#' Rename Standard-Shape Predictions Back to Their Output-Suffixed Form
#'
#' @description
#' Inverse of the renaming `predict.kerasnip_output_view()` does: maps
#' `.pred`/`.pred_class`/`.pred_lower`/`.pred_upper`/`.pred_<level>` back to
#' `.pred_<output>`/`.pred_class_<output>`/`.pred_lower_<output>`/
#' `.pred_upper_<output>`/`.pred_<output>_<level>`, so tailor-adjusted
#' predictions can be spliced back into the full multi-output prediction
#' tibble.
#'
#' @param preds A tibble in the standard single-output prediction shape.
#' @param output A string, the output name to suffix columns with.
#' @return `preds` with its columns renamed to the `_<output>`-suffixed
#'   form.
#' @keywords internal
#' @noRd
kerasnip_rename_view_columns_back <- function(preds, output) {
  nm <- names(preds)
  new_nm <- vapply(
    nm,
    function(n) {
      if (n == ".pred") {
        return(paste0(".pred_", output))
      }
      if (n == ".pred_class") {
        return(paste0(".pred_class_", output))
      }
      if (n == ".pred_lower") {
        return(paste0(".pred_lower_", output))
      }
      if (n == ".pred_upper") {
        return(paste0(".pred_upper_", output))
      }
      if (startsWith(n, ".pred_")) {
        level <- substring(n, nchar(".pred_") + 1)
        return(paste0(".pred_", output, "_", level))
      }
      n
    },
    character(1),
    USE.NAMES = FALSE
  )
  names(preds) <- new_nm
  preds
}

#' Name of the Forecasted-Value Column in One Step's Nested Tibble
#'
#' @description
#' The column inside one row's nested step tibble that holds the forecasted
#' value for `var` (`".pred_<var>"` if the model forecasts more than one
#' variable, `".pred"` otherwise) — mirrors the lookup in
#' `predict.kerasnip_step_view()`/`kerasnip_step_extract()`.
#'
#' @param step_tbl A one-row tibble, a single forecast step's slice of one
#'   sample's nested `.pred` tibble.
#' @param var A string, the forecasted variable, or `NULL` if the model
#'   forecasts only one.
#' @return A string, the column name to read/write.
#' @keywords internal
#' @noRd
kerasnip_step_var_col <- function(step_tbl, var) {
  candidate <- if (!is.null(var)) paste0(".pred_", var) else NA_character_
  if (!is.na(candidate) && candidate %in% names(step_tbl)) {
    return(candidate)
  }
  ".pred"
}

#' Attach a `tailor` Post-Processor to One Output or Step of a Multi-Output
#' or Multistep Workflow
#'
#' @description
#' `workflows::add_tailor()` cannot be used on a kerasnip multi-output or
#' multistep workflow: `tailor::fit()` selects `outcome`/`estimate` via
#' `[[`, which requires exactly one, flat, numeric column, and both a
#' multi-output recipe (`output_1 + output_2 ~ .`) and a multistep model's
#' nested `.pred` list-column violate that (see
#' `vignette("multi_output_postprocessing")`). `kerasnip_add_tailor()` is a
#' kerasnip-owned analogue that attaches a `tailor` post-processor to a
#' single named output or forecast step, using [kerasnip_output_view()] or
#' [kerasnip_step_view()] internally.
#'
#' @details
#' At `fit()` time, the underlying model is trained as usual; the relevant
#' view is then used to fit the `tailor` against that output's/step's
#' predictions (on `data_calibration` if supplied, otherwise on `data`,
#' mirroring `workflows::add_tailor()`'s data-usage convention). At
#' `predict()` time, the full prediction is generated, the target
#' output's/step's value(s) are replaced with the tailor-adjusted values,
#' and everything else (other outputs; other steps in the same nested
#' tibble) is left untouched.
#'
#' Exactly one of `output` or `step` must be supplied: `output` for a
#' multi-output model, `step` (and `var`, if more than one variable is
#' forecast) for a multistep model.
#'
#' @param x An **unfitted** `workflow` whose model has more than one outcome
#'   (multi-output) or is a multistep forecasting model.
#' @param tailor A `tailor::tailor()` specification.
#' @param output A string, the name of the outcome column to post-process
#'   (multi-output models).
#' @param step An integer, the forecast step to post-process (multistep
#'   models).
#' @param var A string, the forecasted variable to post-process; only
#'   needed with `step` if the model forecasts more than one variable.
#' @return A `kerasnip_tailored_workflow`, to be trained with `fit()`.
#' @examples
#' \dontrun{
#' tlr <- tailor::tailor() |> tailor::adjust_numeric_calibration()
#'
#' # multi-output
#' tailored_wf <- kerasnip_add_tailor(wf, tlr, output = "output_1")
#'
#' # multistep
#' tailored_wf <- kerasnip_add_tailor(wf, tlr, step = 2)
#'
#' fit_obj <- fit(tailored_wf, data = train_data, data_calibration = cal_data)
#' predict(fit_obj, new_data = test_data)
#' }
#' @export
kerasnip_add_tailor <- function(
  x,
  tailor,
  output = NULL,
  step = NULL,
  var = NULL
) {
  rlang::check_installed(c("workflows", "tailor"))
  if (!inherits(x, "workflow")) {
    rlang::abort("`x` must be a `workflow`.")
  }
  if (is.null(output) == is.null(step)) {
    rlang::abort("Exactly one of `output` or `step` must be supplied.")
  }
  structure(
    list(
      workflow = x,
      tailor = tailor,
      output = output,
      step = step,
      var = var
    ),
    class = "kerasnip_tailored_workflow"
  )
}

#' Fit Method for `kerasnip_tailored_workflow` Objects
#'
#' @description
#' Trains the underlying multi-output/multistep workflow, then fits
#' `object$tailor` against the target output's/step's predictions (via
#' [kerasnip_output_view()]/[kerasnip_step_view()]) on `data_calibration` if
#' supplied, otherwise on `data`.
#'
#' @param object A `kerasnip_tailored_workflow`, from
#'   [kerasnip_add_tailor()].
#' @param data The training data.
#' @param ... Passed to `fit()` on the underlying workflow.
#' @param data_calibration Optional calibration data for the `tailor`;
#'   defaults to `data` if not supplied.
#' @return A `kerasnip_tailored_fit`, to be used with `predict()`.
#' @keywords internal
#' @exportS3Method generics::fit
fit.kerasnip_tailored_workflow <- function(
  object,
  data,
  ...,
  data_calibration = NULL
) {
  fit_obj <- fit(object$workflow, data = data, ...)
  is_step <- !is.null(object$step)
  view <- if (is_step) {
    kerasnip_step_view(fit_obj, object$step, object$var)
  } else {
    kerasnip_output_view(fit_obj, object$output)
  }

  cal_data <- if (is.null(data_calibration)) data else data_calibration

  if (is_step) {
    cal_preds <- predict(view, new_data = cal_data, type = "numeric")
    truth <- kerasnip_step_truth(view, cal_data)
    n_dropped <- nrow(cal_data) - nrow(cal_preds)
    cal_data_aligned <- cal_data[(n_dropped + 1):nrow(cal_data), , drop = FALSE]
    truth_col <- stats::setNames(list(truth), view$outcome_col)
    cal_preds <- dplyr::bind_cols(
      cal_preds,
      tibble::as_tibble(truth_col),
      cal_data_aligned
    )
    outcome_sym <- rlang::sym(view$outcome_col)
  } else {
    cal_preds <- kerasnip_view_predictions(view, cal_data)
    cal_preds <- dplyr::bind_cols(cal_preds, cal_data)
    outcome_sym <- rlang::sym(object$output)
  }

  tailor_fit <- fit(
    object$tailor,
    cal_preds,
    outcome = !!outcome_sym,
    estimate = dplyr::any_of(c(".pred", ".pred_class")),
    probabilities = c(
      dplyr::contains(".pred_"),
      -dplyr::matches("^\\.pred$|^\\.pred_class$")
    )
  )

  structure(
    list(
      fit_obj = fit_obj,
      output = object$output,
      step = object$step,
      var = object$var,
      tailor_fit = tailor_fit,
      mode = if (is_step) "regression" else view$mode
    ),
    class = "kerasnip_tailored_fit"
  )
}

#' Predict Method for `kerasnip_tailored_fit` Objects
#'
#' @description
#' Predicts from the underlying full workflow, applies the fitted `tailor`
#' to the target output's/step's predictions, and splices the adjusted
#' values back in: `.pred_<output>`/`.pred_class_<output>`-suffixed columns
#' for a multi-output model, or the matching `.step` entry in every row's
#' nested tibble for a multistep model. Every other output/step is returned
#' exactly as a plain `predict()` on the underlying fit would give it.
#'
#' @param object A `kerasnip_tailored_fit`, from `fit()` on a
#'   `kerasnip_tailored_workflow`.
#' @param new_data A data frame of predictors.
#' @param ... Not used.
#' @return A tibble in the full multi-output/multistep prediction shape.
#' @keywords internal
#' @exportS3Method stats::predict
predict.kerasnip_tailored_fit <- function(object, new_data, ...) {
  if (!is.null(object$step)) {
    view <- kerasnip_step_view(object$fit_obj, object$step, object$var)
    view_preds <- predict(view, new_data = new_data, type = "numeric")
    adjusted <- predict(object$tailor_fit, view_preds)

    full_preds <- predict(object$fit_obj, new_data = new_data)
    full_preds$.pred <- purrr::map2(
      full_preds$.pred,
      adjusted$.pred,
      function(step_tbl, new_val) {
        idx <- which(step_tbl$.step == object$step)
        col <- kerasnip_step_var_col(step_tbl, object$var)
        step_tbl[[col]][idx] <- new_val
        step_tbl
      }
    )
    return(full_preds)
  }

  view <- kerasnip_output_view(object$fit_obj, object$output)
  view_preds <- kerasnip_view_predictions(view, new_data)

  adjusted <- predict(object$tailor_fit, view_preds)
  adjusted_renamed <- kerasnip_rename_view_columns_back(adjusted, object$output)

  full_type <- if (object$mode == "classification") "prob" else "numeric"
  full_preds <- predict(object$fit_obj, new_data = new_data, type = full_type)
  if (object$mode == "classification") {
    full_class <- predict(object$fit_obj, new_data = new_data, type = "class")
    full_preds <- dplyr::bind_cols(full_class, full_preds)
  }

  full_preds[names(adjusted_renamed)] <- adjusted_renamed
  full_preds
}
