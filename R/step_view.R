#' View a Single Forecast Step of a Multistep kerasnip Fit
#'
#' @description
#' A kerasnip multistep (vector-valued) regression model returns a nested
#' `.pred` list-column: one inner tibble per row, with a `.step` column plus
#' one prediction column per forecasted variable. `tailor`/`probably` expect
#' a single flat numeric `.pred` column instead — `tailor::check_variable_type()`
#' requires `is.numeric()` on the outcome/estimate columns, which a
#' list-column fails outright.
#'
#' `kerasnip_step_view()` wraps a fitted multistep workflow together with
#' one forecast step (and, if more than one variable is forecast, which
#' variable), presenting it as an ordinary single-output fit: `predict()`
#' returns a flat `.pred` column for that step alone.
#'
#' @details
#' Unlike [kerasnip_output_view()], a multistep model's per-step outcome
#' columns (e.g. `lead_1_value`) are recipe-*engineered* from a single raw
#' column via `step_lead()` — they are not present in a user's raw data the
#' way genuine multi-output columns are. [kerasnip_step_truth()] recovers
#' the true future value at a given step by re-baking the fitted recipe on
#' raw data, which is what [int_conformal_split()][probably::int_conformal_split]
#' uses internally for this class.
#'
#' `probably::int_conformal_full()` is not supported for step views: its
#' refit-per-candidate design would need to substitute a candidate value
#' into the single *raw* column that `step_lead()` derives every step's
#' truth from, which then shifts every nearby row's target — a materially
#' different (and more involved) problem than the single-column substitution
#' [kerasnip_output_view()] uses, and is not implemented here.
#'
#' @param x A fitted (trained) `workflow` whose model is a multistep
#'   regression model (see `create_keras_sequential_spec()`/
#'   `create_keras_functional_spec()` with a vector-valued output).
#' @param step An integer, the forecast step to view.
#' @param var A string, the forecasted variable to view. Required only if
#'   the model forecasts more than one variable; inferred otherwise.
#' @return A `kerasnip_step_view` object.
#' @examples
#' \dontrun{
#' fit_obj <- fit(wf, data = train_data) # a multistep forecasting workflow
#' step_2 <- kerasnip_step_view(fit_obj, step = 2)
#' predict(step_2, new_data = test_data) # -> a single `.pred` column
#' }
#' @export
kerasnip_step_view <- function(x, step, var = NULL) {
  rlang::check_installed(c("workflows", "hardhat"))
  if (!inherits(x, "workflow") || !isTRUE(x$trained)) {
    rlang::abort("`x` must be a fitted `workflow`.")
  }

  fit_parsnip <- workflows::extract_fit_parsnip(x)
  multistep_info <- fit_parsnip$fit$multistep_info
  if (is.null(multistep_info)) {
    rlang::abort(c(
      "`x` does not look like a multistep forecasting model.",
      i = "Use kerasnip_output_view() for genuine multi-output models instead."
    ))
  }

  uniq_vars <- unique(multistep_info$vars)
  if (is.null(var)) {
    if (length(uniq_vars) > 1) {
      rlang::abort(paste0(
        "Multiple forecasted variables found (",
        paste0("`", uniq_vars, "`", collapse = ", "),
        "); specify `var`."
      ))
    }
    var <- uniq_vars
  } else if (!isTRUE(var %in% uniq_vars)) {
    rlang::abort(paste0(
      "`var` must be one of ",
      paste0("`", uniq_vars, "`", collapse = ", "),
      ", not `",
      var,
      "`."
    ))
  }
  if (!isTRUE(step %in% multistep_info$steps)) {
    rlang::abort(paste0(
      "`step` must be one of ",
      paste0(multistep_info$steps, collapse = ", "),
      ", not ",
      step,
      "."
    ))
  }

  mold <- hardhat::extract_mold(x)
  outcome_names <- names(mold$outcomes)
  idx <- which(multistep_info$steps == step & multistep_info$vars == var)
  if (length(idx) != 1) {
    rlang::abort(paste0(
      "Could not find a unique outcome column for step ",
      step,
      ", variable `",
      var,
      "`."
    ))
  }

  structure(
    list(workflow = x, step = step, var = var, outcome_col = outcome_names[idx]),
    class = "kerasnip_step_view"
  )
}

#' Read One Forecast Step's Prediction Column from a Step Tibble
#'
#' @description
#' Given `row`, one forecast step's row from a multistep model's nested
#' `.pred` tibble, resolves and returns the value for `prefix` (`".pred"`,
#' `".pred_lower"`, or `".pred_upper"`): the `<prefix>_<var>` column if the
#' model forecasts more than one variable, `prefix` itself otherwise.
#'
#' @param row A one-row tibble, a single forecast step's slice of one
#'   sample's nested `.pred` tibble.
#' @param var A string, the forecasted variable to read, or `NULL` if the
#'   model forecasts only one.
#' @param prefix One of `".pred"`, `".pred_lower"`, `".pred_upper"`.
#' @return A length-1 numeric value.
#' @keywords internal
#' @noRd
kerasnip_step_extract <- function(row, var, prefix) {
  candidate <- if (!is.null(var)) paste0(prefix, "_", var) else NA_character_
  if (!is.na(candidate) && candidate %in% names(row)) {
    return(row[[candidate]])
  }
  if (prefix %in% names(row)) {
    return(row[[prefix]])
  }
  rlang::abort(paste0(
    "Could not find a `", prefix, "` column for variable `", var, "`."
  ))
}

#' Predict Method for `kerasnip_step_view` Objects
#'
#' @description
#' Predicts from the wrapped multistep workflow, then extracts
#' `object$step`'s (and, if set, `object$var`'s) value from every row's
#' nested `.pred` tibble into a flat column, so the result reads like a
#' single-output `predict()` call.
#'
#' @param object A `kerasnip_step_view`.
#' @param new_data A data frame of predictors.
#' @param type One of `"numeric"`, `"conf_int"`, or `"pred_int"`.
#' @param ... Passed to `predict()` on the wrapped workflow.
#' @return A tibble with a `.pred` column (`"numeric"`), or `.pred`/
#'   `.pred_lower`/`.pred_upper` (`"conf_int"`/`"pred_int"`).
#' @keywords internal
#' @exportS3Method stats::predict
predict.kerasnip_step_view <- function(object, new_data, type = "numeric", ...) {
  preds <- predict(object$workflow, new_data = new_data, type = type, ...)

  prefixes <- switch(
    type,
    numeric = c(pred = ".pred"),
    conf_int = ,
    pred_int = c(pred = ".pred", lower = ".pred_lower", upper = ".pred_upper"),
    rlang::abort(paste0("Unsupported `type` for a step view: `", type, "`."))
  )

  cols <- lapply(prefixes, function(prefix) {
    vapply(
      preds$.pred,
      function(step_tbl) {
        row <- step_tbl[step_tbl$.step == object$step, , drop = FALSE]
        if (nrow(row) != 1) {
          rlang::abort(paste0(
            "Step ", object$step, " not present in the forecast horizon."
          ))
        }
        kerasnip_step_extract(row, object$var, prefix)
      },
      numeric(1)
    )
  })
  names(cols) <- paste0(".pred_", names(prefixes))
  names(cols)[names(prefixes) == "pred"] <- ".pred"

  tibble::as_tibble(cols)
}

#' Recover Truth Values for a Multistep Forecast Step
#'
#' @description
#' A multistep model's per-step outcome columns (e.g. `lead_2_value`) are
#' engineered by `step_lead()` from a single raw column, so they are not
#' present in a user's raw data the way genuine multi-output columns are.
#' This re-bakes the fitted recipe on `new_data` to recover the actual
#' future value at [kerasnip_step_view()]'s step, for calibration/interval
#' use. Rows too close to the end of `new_data` for the lead to be computed
#' return `NA` (dropped automatically by calibration routines that call
#' `sort()`/`stats::complete.cases()` on the result).
#'
#' @param view A `kerasnip_step_view`.
#' @param new_data A data frame of raw predictors (and the original outcome
#'   column `step_lead()` was applied to).
#' @return A numeric vector, one truth value per row of `new_data`.
#' @export
kerasnip_step_truth <- function(view, new_data) {
  rlang::check_installed(c("workflows", "recipes"))
  rec <- workflows::extract_recipe(view$workflow, estimated = TRUE)
  baked <- recipes::bake(rec, new_data = new_data, recipes::all_outcomes())
  baked[[view$outcome_col]]
}

#' Augment Method for `kerasnip_step_view` Objects
#'
#' @description
#' Binds `predict(x, new_data, type = "numeric")`'s `.pred` column with the
#' step's truth (from [kerasnip_step_truth()]) and `new_data`, mirroring
#' `workflows:::augment.workflow()`. Used internally by
#' `int_conformal_split.kerasnip_step_view()`. Rows `step_sequence()` drops
#' for lacking a full window of history are dropped here too, to stay
#' aligned with `predict()`'s row count.
#'
#' @param x A `kerasnip_step_view`.
#' @param new_data A data frame of raw predictors (and the original outcome
#'   column `step_lead()` was applied to).
#' @param ... Not used.
#' @return A tibble: `.pred`, the step's truth column (named
#'   `x$outcome_col`), and `new_data`'s columns, aligned to the rows that
#'   survived windowing.
#' @keywords internal
#' @exportS3Method generics::augment
augment.kerasnip_step_view <- function(x, new_data, ...) {
  preds <- predict(x, new_data = new_data, type = "numeric")
  truth <- kerasnip_step_truth(x, new_data)
  truth_col <- stats::setNames(list(truth), x$outcome_col)

  # step_sequence() drops leading rows lacking a full window of history, so
  # predict()/bake() can return fewer rows than `new_data`; align on the
  # trailing rows that survived (the drop is always from the start).
  n_dropped <- nrow(new_data) - nrow(preds)
  aligned_new_data <- new_data[(n_dropped + 1):nrow(new_data), , drop = FALSE]

  dplyr::bind_cols(preds, tibble::as_tibble(truth_col), aligned_new_data)
}

#' Split Conformal Inference Method for `kerasnip_step_view` Objects
#'
#' @description
#' Calibration-set conformal intervals for one forecast step of a multistep
#' fit. Mirrors `probably`'s own (private) `int_conformal_split.workflow()`,
#' using only `generics::augment()` (implemented for this class via
#' [kerasnip_step_truth()]), rather than `probably`'s unexported internals.
#'
#' @param object A `kerasnip_step_view`.
#' @param cal_data A data frame of raw calibration predictors (and the
#'   original outcome column `step_lead()` was applied to).
#' @param ... Not used.
#' @return A `conformal_reg_split`/`int_conformal_split` object; `predict()`
#'   on it (from `probably`) works unmodified, since it dispatches back to
#'   `predict.kerasnip_step_view()`.
#' @keywords internal
#' @exportS3Method probably::int_conformal_split
int_conformal_split.kerasnip_step_view <- function(object, cal_data, ...) {
  rlang::check_dots_empty()
  y_name <- object$outcome_col
  cal_pred <- generics::augment(object, cal_data)
  cal_pred$.resid <- cal_pred[[y_name]] - cal_pred$.pred
  res <- list(
    resid = sort(abs(cal_pred$.resid)),
    wflow = object,
    n = sum(!is.na(cal_pred$.resid))
  )
  class(res) <- c("conformal_reg_split", "int_conformal_split")
  res
}
