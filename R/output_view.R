#' View a Single Output of a Multi-Output kerasnip Fit
#'
#' @description
#' `tailor` and `probably` are built around models with a single outcome
#' column and a single `.pred`/`.pred_class` prediction column. A kerasnip
#' multi-output model (e.g. a recipe with `output_1 + output_2 ~ .`) instead
#' produces `.pred_output_1`, `.pred_output_2`, ... columns from multiple
#' truth columns in one fit — the standard `parsnip::maybe_multivariate()`
#' shape, but one `tailor::fit()`/`workflows::add_tailor()` call cannot
#' consume it (it selects `outcome`/`estimate` via `[[`, which requires
#' exactly one column).
#'
#' `kerasnip_output_view()` wraps a fitted multi-output workflow together
#' with one output name, presenting it as if it were an ordinary
#' single-output fit: `predict()` returns standard `.pred` / `.pred_class` /
#' `.pred_<level>` columns for that output alone, letting you calibrate or
#' post-process each output separately with the usual `tailor`/`probably`
#' calls (see `vignette("multi_output_postprocessing")`).
#'
#' @param x A fitted (trained) `workflow` whose model has more than one
#'   outcome column.
#' @param output A string, the name of the outcome column to view.
#' @return A `kerasnip_output_view` object.
#' @examples
#' \dontrun{
#' fit_obj <- fit(wf, data = train_data) # wf predicts output_1 and output_2
#' view_1 <- kerasnip_output_view(fit_obj, "output_1")
#' predict(view_1, new_data = test_data) # -> a single `.pred` column
#' }
#' @export
kerasnip_output_view <- function(x, output) {
  rlang::check_installed(c("workflows", "hardhat"))
  if (!inherits(x, "workflow") || !isTRUE(x$trained)) {
    rlang::abort("`x` must be a fitted `workflow`.")
  }

  mold <- hardhat::extract_mold(x)
  outcome_names <- names(mold$outcomes)

  if (length(outcome_names) < 2) {
    rlang::abort(c(
      "`x` does not look like a multi-output model.",
      i = paste0(
        "It has a single outcome (`", outcome_names, "`); ",
        "use its predictions directly instead of a view."
      )
    ))
  }
  if (!isTRUE(output %in% outcome_names)) {
    rlang::abort(paste0(
      "`output` must be one of ",
      paste0("`", outcome_names, "`", collapse = ", "),
      ", not `",
      output,
      "`."
    ))
  }

  fit_parsnip <- workflows::extract_fit_parsnip(x)
  structure(
    list(workflow = x, output = output, mode = fit_parsnip$spec$mode),
    class = "kerasnip_output_view"
  )
}

#' Predict Method for `kerasnip_output_view` Objects
#'
#' @description
#' Predicts from the wrapped multi-output workflow, then selects and renames
#' `object$output`'s columns down to the standard single-output shape
#' (`.pred`, `.pred_class`, `.pred_<level>`, or `.pred`/`.pred_lower`/
#' `.pred_upper`), so the result reads like a single-output `predict()` call.
#'
#' @param object A `kerasnip_output_view`.
#' @param new_data A data frame of predictors.
#' @param type One of `"numeric"`, `"class"`, `"prob"`, `"conf_int"`, or
#'   `"pred_int"`. Defaults to `"class"` for a classification view, `
#'   "numeric"` otherwise.
#' @param ... Passed to `predict()` on the wrapped workflow.
#' @return A tibble in the standard single-output prediction shape.
#' @keywords internal
#' @exportS3Method stats::predict
predict.kerasnip_output_view <- function(object, new_data, type = NULL, ...) {
  if (is.null(type)) {
    type <- if (object$mode == "classification") "class" else "numeric"
  }
  preds <- predict(object$workflow, new_data = new_data, type = type, ...)
  output <- object$output

  out <- switch(
    type,
    numeric = {
      col <- preds[paste0(".pred_", output)]
      names(col) <- ".pred"
      col
    },
    class = {
      col <- preds[paste0(".pred_class_", output)]
      names(col) <- ".pred_class"
      col
    },
    prob = {
      prefix <- paste0(".pred_", output, "_")
      prob_cols <- grep(paste0("^", prefix), names(preds), value = TRUE)
      if (length(prob_cols) == 0) {
        rlang::abort(paste0(
          "Could not find probability columns for output `", output, "`."
        ))
      }
      cols <- preds[prob_cols]
      names(cols) <- paste0(".pred_", substring(prob_cols, nchar(prefix) + 1))
      cols
    },
    conf_int = ,
    pred_int = {
      cols <- c(
        paste0(".pred_", output),
        paste0(".pred_lower_", output),
        paste0(".pred_upper_", output)
      )
      cols <- cols[cols %in% names(preds)]
      out <- preds[cols]
      names(out) <- sub(paste0("_", output, "$"), "", names(out))
      out
    },
    rlang::abort(paste0("Unsupported `type`: `", type, "`."))
  )

  tibble::as_tibble(out)
}

#' Extract Mold Method for `kerasnip_output_view` Objects
#'
#' @description
#' Returns the wrapped workflow's mold with `$outcomes` (and, if present,
#' `$blueprint$ptypes$outcomes`) sliced down to `x$output`'s single column,
#' so downstream code that reads `names(extract_mold(x)$outcomes)` (such as
#' `probably::int_conformal_split()`) sees a single-outcome fit.
#'
#' @param x A `kerasnip_output_view`.
#' @param ... Not used.
#' @return A `hardhat` mold with a single outcome column.
#' @keywords internal
#' @exportS3Method hardhat::extract_mold
extract_mold.kerasnip_output_view <- function(x, ...) {
  mold <- hardhat::extract_mold(x$workflow)
  mold$outcomes <- mold$outcomes[x$output]
  if (!is.null(mold$blueprint$ptypes$outcomes)) {
    mold$blueprint$ptypes$outcomes <- mold$blueprint$ptypes$outcomes[x$output]
  }
  mold
}

#' Augment Method for `kerasnip_output_view` Objects
#'
#' @description
#' Binds `predict(x, new_data, type = "numeric")`'s `.pred` column with
#' `new_data`, mirroring `workflows:::augment.workflow()`. Used internally by
#' `int_conformal_split.kerasnip_output_view()`, which needs both the
#' prediction and `new_data`'s truth column in one data frame.
#'
#' @param x A `kerasnip_output_view`.
#' @param new_data A data frame of predictors, including `x$output`'s truth
#'   column.
#' @param ... Not used.
#' @return `new_data` with a `.pred` column prepended.
#' @keywords internal
#' @exportS3Method generics::augment
augment.kerasnip_output_view <- function(x, new_data, ...) {
  preds <- predict(x, new_data = new_data, type = "numeric")
  dplyr::bind_cols(preds, new_data)
}

#' Split Conformal Inference Method for `kerasnip_output_view` Objects
#'
#' @description
#' Calibration-set conformal intervals for one output of a multi-output fit.
#' Mirrors `probably`'s own (private) `int_conformal_split.workflow()`, using
#' only `hardhat::extract_mold()` and `generics::augment()` (both implemented
#' for this class), rather than `probably`'s unexported internals.
#'
#' @param object A `kerasnip_output_view`.
#' @param cal_data A data frame of calibration predictors and truth.
#' @param ... Not used.
#' @return A `conformal_reg_split`/`int_conformal_split` object; `predict()`
#'   on it (from `probably`) works unmodified, since it dispatches back to
#'   `predict.kerasnip_output_view()`.
#' @keywords internal
#' @exportS3Method probably::int_conformal_split
int_conformal_split.kerasnip_output_view <- function(object, cal_data, ...) {
  rlang::check_dots_empty()
  y_name <- object$output
  cal_pred <- generics::augment(object, cal_data)
  cal_pred$.resid <- cal_pred[[y_name]] - cal_pred$.pred
  res <- list(
    resid = sort(abs(cal_pred$.resid)),
    wflow = object,
    n = nrow(cal_pred)
  )
  class(res) <- c("conformal_reg_split", "int_conformal_split")
  res
}

# int_conformal_full() for a single output of a multi-output fit
#
# `probably::int_conformal_full()` refits the model once per candidate value
# of every new observation, which only makes sense for a single-outcome fit:
# every one of its internal helpers (`get_outcome_name()`, `var_model()`,
# `grid_one()`, ...) assumes there is exactly one outcome and reads
# `x$pre$mold$blueprint$ptypes$outcomes` directly, bypassing the
# `hardhat::extract_mold()` generic entirely. That means a lightweight view
# object cannot reuse those (unexported) helpers, and — more fundamentally —
# refitting a *jointly* trained multi-head model for a candidate value of
# ONE output leaves the other output(s) without a real training target for
# that synthetic row.
#
# This implementation is kerasnip's own, using only public generics
# (predict()/fit()/augment() on the view), not probably's private internals.
# For the missing "other output(s) on the synthetic row" problem, it
# substitutes the model's own current point-prediction as a self-consistent
# placeholder: because the placeholder equals what that head already
# predicts, its loss contribution for that single row is ~zero, so the
# other head(s) should not be measurably disturbed while the target head
# still responds to the candidate value under test. This is a reasonable,
# but unproven, choice — treat the resulting intervals accordingly.
#
# Only `control$method = "grid"` is supported (the same restriction this
# package's own single-output conformal tests already use); "iterative"
# requires probably's private root-finding helpers and is out of scope.

#' Outcome Column Names of a Fitted Workflow
#'
#' @description
#' Thin wrapper around `hardhat::extract_mold()` returning just the outcome
#' column names, single- or multi-output.
#'
#' @param x A fitted `workflow`.
#' @return A character vector of outcome column names.
#' @keywords internal
#' @noRd
kerasnip_mold_outcome_names <- function(x) {
  names(hardhat::extract_mold(x)$outcomes)
}

#' Fit the Residual-Variance Model for `int_conformal_full()`
#'
#' @description
#' Fits an `mgcv::gam()` predicting squared training residuals from the
#' point prediction, used to size the per-observation candidate-value search
#' range in `kerasnip_setup_new_data()`. Mirrors `probably`'s own (private)
#' `var_model()`, just built from `view`'s predictions/truth directly instead
#' of `probably`'s `get_outcome_name()`.
#'
#' @param view A `kerasnip_output_view`.
#' @param train_data The training data used to fit `view`'s underlying model.
#' @return A fitted `mgcv::gam()` object.
#' @keywords internal
#' @noRd
kerasnip_var_model <- function(view, train_data) {
  rlang::check_installed("mgcv")
  y_name <- view$output
  train_res <- predict(view, new_data = train_data)
  train_res$resid <- train_data[[y_name]] - train_res$.pred
  train_res$sq <- train_res$resid^2
  var_mod <- try(
    mgcv::gam(sq ~ s(.pred), data = train_res, family = stats::Gamma(link = "log")),
    silent = TRUE
  )
  if (inherits(var_mod, "try-error")) {
    rlang::abort(c(
      "The model to estimate the possible interval length failed.",
      i = conditionMessage(attr(var_mod, "condition"))
    ))
  }
  var_mod
}

#' Per-Observation Candidate-Value Search Bound for `int_conformal_full()`
#'
#' @description
#' Predicts each row of `new_data` and attaches a `.bound` column, the
#' half-width of the candidate-value grid `kerasnip_grid_one_output_view()`
#' will search, sized from `view$.var_model`'s residual-variance estimate.
#'
#' @param view A `kerasnip_output_view` with a `.var_model` element (set by
#'   `int_conformal_full.kerasnip_output_view()`).
#' @param new_data A data frame of predictors to predict on.
#' @param multiplier Scalar, `control_conformal_full()$var_multiplier`.
#' @return `predict(view, new_data)` with a `.bound` column added.
#' @keywords internal
#' @noRd
kerasnip_setup_new_data <- function(view, new_data, multiplier) {
  new_pred <- predict(view, new_data = new_data)
  var_pred <- sqrt(as.vector(
    predict(view$.var_model, new_pred, type = "response")
  ))
  new_pred$.bound <- multiplier * var_pred
  new_pred
}

#' Resolve the Conformal Interval from Trial-Fit Differences
#'
#' @description
#' Given one row per candidate value (as built by
#' `kerasnip_trial_fit_output_view()`), finds the widest candidate range
#' whose residual stays within the training-residual quantile. Mirrors
#' `probably`'s own (private) `compute_bound()`.
#'
#' @param x A tibble with `trial` and `difference` columns, one row per
#'   candidate value tried.
#' @param predicted Scalar, the observation's original point prediction.
#' @return A one-row tibble with `.pred_lower`/`.pred_upper` (`NA` if no
#'   bound could be determined).
#' @keywords internal
#' @noRd
kerasnip_compute_bound <- function(x, predicted) {
  x <- x[stats::complete.cases(x), ]
  if (nrow(x) == 0 || all(x$difference < 0)) {
    warning("Could not determine bounds.", call. = FALSE)
    return(tibble::tibble(.pred_lower = NA_real_, .pred_upper = NA_real_))
  }
  upper <- x[x$trial >= predicted & x$difference >= 0, ]
  upper <- if (nrow(upper) > 0) min(upper$trial) else NA_real_
  lower <- x[x$trial <= predicted & x$difference >= 0, ]
  lower <- if (nrow(lower) > 0) max(lower$trial) else NA_real_
  tibble::tibble(.pred_lower = lower, .pred_upper = upper)
}

#' Refit and Score One Candidate Value for `int_conformal_full()`
#'
#' @description
#' Sets the target output's value on the last (synthetic) row of
#' `trial_data` to `trial`, refits `view`'s workflow from scratch, and
#' compares the synthetic row's residual to the quantile of the (unchanged)
#' training rows' residuals under the refit model. Mirrors `probably`'s own
#' (private) `trial_fit()`.
#'
#' @param trial Scalar, the candidate outcome value to test.
#' @param trial_data Training data plus one synthetic row (real predictors,
#'   placeholder value(s) for any other output(s), `NA` for the target
#'   output prior to this call).
#' @param view A `kerasnip_output_view` (only used for `view$workflow`; a
#'   fresh view of the refit model is built to compute predictions).
#' @param level The conformal level, passed to `stats::quantile()`.
#' @param y_name The target output's column name.
#' @return A one-row tibble with `quantile`, `trial`, `.abs_resid`, and
#'   `difference` (`.abs_resid - quantile`); `NA` columns if the refit
#'   itself failed.
#' @keywords internal
#' @noRd
kerasnip_trial_fit_output_view <- function(trial, trial_data, view, level, y_name) {
  trial_data[[y_name]][nrow(trial_data)] <- trial
  tmp_fit <- try(fit(view$workflow, trial_data), silent = TRUE)
  if (inherits(tmp_fit, "try-error")) {
    return(tibble::tibble(quantile = NA_real_, trial = trial, .abs_resid = NA_real_))
  }
  tmp_view <- kerasnip_output_view(tmp_fit, y_name)
  tmp_preds <- predict(tmp_view, new_data = trial_data)
  abs_resid <- abs(trial_data[[y_name]] - tmp_preds$.pred)
  quant_val <- stats::quantile(abs_resid[-length(abs_resid)], probs = level)
  res <- tibble::tibble(
    quantile = unname(quant_val),
    trial = trial,
    .abs_resid = abs_resid[length(abs_resid)]
  )
  res$difference <- res$.abs_resid - res$quantile
  res
}

#' Grid-Search Conformal Interval for One New Observation
#'
#' @description
#' Builds the candidate-value grid for one new observation (row of
#' `predict(view, new_data)` plus `.bound`), refits the model for every
#' candidate via `kerasnip_trial_fit_output_view()`, and resolves the
#' interval via `kerasnip_compute_bound()`. The other output(s)' value(s)
#' for this synthetic row are set to the current model's own point
#' prediction (see the design note above `kerasnip_mold_outcome_names()`).
#' Mirrors `probably`'s own (private) `grid_one()`.
#'
#' @param new_data_row A one-row data frame: predictors plus `.pred`/`.bound`
#'   from `kerasnip_setup_new_data()`.
#' @param view A `kerasnip_output_view`.
#' @param train_data The training data used to fit `view`'s underlying model.
#' @param level The conformal level.
#' @param ctrl A `probably::control_conformal_full()` object.
#' @return A one-row tibble with `.pred_lower`/`.pred_upper`.
#' @keywords internal
#' @noRd
kerasnip_grid_one_output_view <- function(new_data_row, view, train_data, level, ctrl) {
  y_name <- view$output
  pred_val <- new_data_row$.pred
  bound <- new_data_row$.bound
  row_predictors <- new_data_row[, setdiff(names(new_data_row), c(".pred", ".bound")), drop = FALSE]

  # Placeholder for the other output(s): see the design note above.
  full_preds <- predict(view$workflow, new_data = row_predictors)
  other_outputs <- setdiff(kerasnip_mold_outcome_names(view$workflow), y_name)
  for (nm in other_outputs) {
    row_predictors[[nm]] <- full_preds[[paste0(".pred_", nm)]]
  }
  row_predictors[[y_name]] <- NA_real_

  trial_data <- dplyr::bind_rows(train_data, row_predictors)
  trial_vals <- seq(pred_val - bound, pred_val + bound, length.out = ctrl$trial_points)
  res <- purrr::map_dfr(
    trial_vals,
    kerasnip_trial_fit_output_view,
    trial_data = trial_data,
    view = view,
    level = level,
    y_name = y_name
  )
  kerasnip_compute_bound(res, pred_val)
}

#' Full Conformal Inference Method for `kerasnip_output_view` Objects
#'
#' @description
#' Full (refit-per-candidate) conformal intervals for one output of a
#' multi-output fit. This is kerasnip's own implementation (see the design
#' note above `kerasnip_mold_outcome_names()`), not a reuse of `probably`'s
#' private internals, since those assume a single-outcome fit throughout.
#' Only `control$method = "grid"` is supported.
#'
#' @param object A `kerasnip_output_view` viewing a regression output.
#' @param train_data The training data used to fit `object`'s underlying
#'   model.
#' @param ... Not used.
#' @param control A `probably::control_conformal_full()` object; defaults to
#'   `method = "grid"` if not supplied.
#' @return A `kerasnip_conformal_full`/`int_conformal_full` object; call
#'   `predict()` on it to get intervals for new data.
#' @keywords internal
#' @exportS3Method probably::int_conformal_full
int_conformal_full.kerasnip_output_view <- function(
  object,
  train_data,
  ...,
  control = NULL
) {
  rlang::check_dots_empty()
  rlang::check_installed("probably")
  if (is.null(control)) {
    control <- probably::control_conformal_full(method = "grid")
  }
  if (!identical(control$method, "grid")) {
    rlang::abort(c(
      "Only `\"grid\"` is supported for a multi-output kerasnip view.",
      i = "Pass `control = probably::control_conformal_full(method = \"grid\")`."
    ))
  }
  if (!identical(object$mode, "regression")) {
    rlang::abort("`object` should view a regression output.")
  }

  var_mod <- kerasnip_var_model(object, train_data)
  object$.var_model <- var_mod
  structure(
    list(wflow = object, training = train_data, control = control),
    class = c("kerasnip_conformal_full", "int_conformal_full")
  )
}

#' Predict Method for `kerasnip_conformal_full` Objects
#'
#' @description
#' Computes full-conformal intervals for `new_data`, one grid search per row
#' via `kerasnip_grid_one_output_view()`.
#'
#' @param object A `kerasnip_conformal_full` object, from
#'   `int_conformal_full.kerasnip_output_view()`.
#' @param new_data A data frame of predictors.
#' @param level The conformal level.
#' @param ... Not used.
#' @return A tibble with `.pred_lower`/`.pred_upper` columns, one row per
#'   row of `new_data`.
#' @keywords internal
#' @exportS3Method stats::predict
predict.kerasnip_conformal_full <- function(object, new_data, level = 0.95, ...) {
  rlang::check_dots_empty()
  new_pred <- kerasnip_setup_new_data(
    object$wflow,
    new_data,
    object$control$var_multiplier
  )
  full <- dplyr::bind_cols(new_data, new_pred)
  new_rows <- split(full, seq_len(nrow(full)))
  purrr::map_dfr(
    new_rows,
    kerasnip_grid_one_output_view,
    view = object$wflow,
    train_data = object$training,
    level = level,
    ctrl = object$control
  )
}
