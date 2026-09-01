#' View a Single Forecast Step of a Multistep kerasnip Fit
#'
#' @description
#' A kerasnip multistep (vector-valued) regression model returns a nested
#' `.pred` list-column: one inner tibble per row, with a `.step` column plus
#' one prediction column per forecasted variable. `tailor`/`probably` expect
#' a single flat numeric `.pred` column instead —
#' `tailor::check_variable_type()` requires `is.numeric()` on the
#' outcome/estimate columns, which a list-column fails outright.
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
#' raw data, which is what
#' [int_conformal_split()][probably::int_conformal_split] uses internally for
#' this class.
#'
#' `probably::int_conformal_full()` is also supported (see
#' `int_conformal_full.kerasnip_step_view()`), with a materially different
#' design from [kerasnip_output_view()]'s: refitting for a candidate value
#' at this step means substituting it into the single *raw* column
#' `step_lead()` derives every step's truth from, which shifts every nearby
#' row's target too. It is only supported when `step_lead()` and
#' `step_sequence()` share a single source column, matching all of
#' kerasnip's own multistep examples.
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
    list(
      workflow = x,
      step = step,
      var = var,
      outcome_col = outcome_names[idx]
    ),
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
    "Could not find a `",
    prefix,
    "` column for variable `",
    var,
    "`."
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
predict.kerasnip_step_view <- function(
  object,
  new_data,
  type = "numeric",
  ...
) {
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
            "Step ",
            object$step,
            " not present in the forecast horizon."
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

# int_conformal_full() for one forecast step of a multistep fit
#
# Unlike kerasnip_output_view() (see the design note in R/output_view.R),
# a multistep model's step targets are not independent raw columns — every
# `lead_k_<var>` column is derived from the *same* single raw column by
# `step_lead()`. Testing a candidate value for one step therefore means
# writing that candidate into the raw column at the appropriate future
# offset, which also supplies (part of) the targets for every *other* step
# forecast from the same origin. Those other steps' placeholder values (the
# current model's own forecast, same idea as kerasnip_output_view()'s
# "other output(s)" placeholder) are written to the raw column at their own
# offsets; the target step's offset is then overwritten with each trial
# value in turn and the whole (now longer) raw series is refit.
#
# This only works when `step_lead()` and `step_sequence()) draw on the same
# single raw column, which is required and validated in
# `int_conformal_full.kerasnip_step_view()`. Only `control$method = "grid"`
# is supported, same restriction as kerasnip_output_view().

kerasnip_step_recipe_step <- function(view, subclass) {
  rlang::check_installed("workflows")
  rec <- workflows::extract_recipe(view$workflow, estimated = TRUE)
  matches <- Filter(function(s) inherits(s, subclass), rec$steps)
  if (length(matches) != 1) {
    rlang::abort(paste0(
      "Could not find a unique `",
      subclass,
      "` step in the recipe."
    ))
  }
  matches[[1]]
}

#' Fit the Residual-Variance Model for a Step View's `int_conformal_full()`
#'
#' @description
#' Fits an `mgcv::gam()` predicting squared training residuals from the
#' point prediction, used to size the per-observation candidate-value search
#' range. Mirrors `kerasnip_var_model()` (`R/output_view.R`), built from a
#' step view's [predict()]/[kerasnip_step_truth()] instead of an output
#' view's `predict()`/raw truth column.
#'
#' @param view A `kerasnip_step_view`.
#' @param train_data The raw training data used to fit `view`'s underlying
#'   model.
#' @return A fitted `mgcv::gam()` object.
#' @keywords internal
#' @noRd
kerasnip_step_var_model <- function(view, train_data) {
  rlang::check_installed("mgcv")
  train_res <- predict(view, new_data = train_data)
  truth <- kerasnip_step_truth(view, train_data)
  train_res$resid <- truth - train_res$.pred
  train_res$sq <- train_res$resid^2
  train_res <- train_res[stats::complete.cases(train_res[c(".pred", "sq")]), ]
  var_mod <- try(
    mgcv::gam(
      sq ~ s(.pred),
      data = train_res,
      family = stats::Gamma(link = "log")
    ),
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

#' Refit and Score One Candidate Value for a Step View's `int_conformal_full()`
#'
#' @description
#' Sets `raw_col` at `target_row_idx` (the future raw row supplying the
#' target step's truth) to `trial`, refits `view`'s workflow on the whole
#' (training + new window + placeholder future) raw series, and compares
#' the target window's residual to the quantile of every other window's
#' residual under the refit model. Mirrors
#' `kerasnip_trial_fit_output_view()` (`R/output_view.R`).
#'
#' @param trial Scalar, the candidate value to test.
#' @param trial_data The full raw series to refit on: training data, the new
#'   observation's window, and `horizon` placeholder future rows (real
#'   value at `target_row_idx` pending this call's substitution).
#' @param view A `kerasnip_step_view` (only used for `view$workflow`,
#'   `view$step`, `view$var`; a fresh view of the refit model is built to
#'   compute predictions/truth).
#' @param level The conformal level, passed to `stats::quantile()`.
#' @param raw_col The raw column `step_lead()`/`step_sequence()` share.
#' @param target_row_idx Integer, `trial_data`'s row index to write `trial`
#'   into.
#' @param target_position Integer, the target window's row index in
#'   `predict()`'s (row-dropped) output.
#' @return A one-row tibble with `quantile`, `trial`, `.abs_resid`, and
#'   `difference` (`.abs_resid - quantile`); `NA` columns if the refit
#'   itself failed.
#' @keywords internal
#' @noRd
kerasnip_trial_fit_step_view <- function(
  trial,
  trial_data,
  view,
  level,
  raw_col,
  target_row_idx,
  target_position
) {
  trial_data[[raw_col]][target_row_idx] <- trial
  tmp_fit <- try(fit(view$workflow, trial_data), silent = TRUE)
  if (inherits(tmp_fit, "try-error")) {
    return(tibble::tibble(
      quantile = NA_real_,
      trial = trial,
      .abs_resid = NA_real_
    ))
  }
  tmp_view <- kerasnip_step_view(tmp_fit, view$step, view$var)
  tmp_preds <- predict(tmp_view, new_data = trial_data)
  truth <- kerasnip_step_truth(tmp_view, trial_data)
  abs_resid <- abs(truth - tmp_preds$.pred)
  quant_val <- stats::quantile(
    abs_resid[-target_position],
    probs = level,
    na.rm = TRUE
  )
  res <- tibble::tibble(
    quantile = unname(quant_val),
    trial = trial,
    .abs_resid = abs_resid[target_position]
  )
  res$difference <- res$.abs_resid - res$quantile
  res
}

#' Grid-Search Conformal Interval for One New Observation (Step View)
#'
#' @description
#' Builds the raw-series augmentation for one new observation (a window
#' ending at `new_data` row `row_idx`'s position): the window's own raw
#' context, plus `horizon` placeholder future rows (the current model's own
#' forecast for every step, at the raw offset each step's `step_lead()`
#' target reads from). Refits the model for every candidate value of the
#' target step via `kerasnip_trial_fit_step_view()`, and resolves the
#' interval via `kerasnip_compute_bound()` (`R/output_view.R`). Assumes
#' `new_data` is a raw continuation of `train_data` (so the combined series
#' is a single valid sequence for `step_lead()`/`step_sequence()`).
#'
#' @param row_idx Integer, the row of `new_pred`/`full_pred` (and the
#'   corresponding window in `new_data`) to build an interval for.
#' @param view A `kerasnip_step_view`.
#' @param train_data The raw training data used to fit `view`'s underlying
#'   model.
#' @param new_data The raw data `new_pred`/`full_pred` were predicted from.
#' @param full_pred `predict(view$workflow, new_data)`: every step's
#'   forecast, for placeholder values at steps other than `view$step`.
#' @param new_pred `kerasnip_setup_new_data(view, new_data, ...)`: `view`'s
#'   own step's point prediction (`.pred`) and search bound (`.bound`).
#' @param level The conformal level.
#' @param ctrl A `probably::control_conformal_full()` object.
#' @param seq_info The fitted `step_sequence()` step (for `$timesteps`).
#' @param raw_col The raw column `step_lead()`/`step_sequence()` share.
#' @return A one-row tibble with `.pred_lower`/`.pred_upper`.
#' @keywords internal
#' @noRd
kerasnip_grid_one_step_view <- function(
  row_idx,
  view,
  train_data,
  new_data,
  full_pred,
  new_pred,
  level,
  ctrl,
  seq_info,
  raw_col
) {
  pred_val <- new_pred$.pred[row_idx]
  bound <- new_pred$.bound[row_idx]
  timesteps <- seq_info$timesteps

  window_end_idx <- timesteps - 1 + row_idx
  window_raw_rows <- new_data[seq_len(window_end_idx), , drop = FALSE]

  step_tbl <- full_pred$.pred[[row_idx]]
  steps <- step_tbl$.step
  var_col <- kerasnip_step_var_col(step_tbl, view$var)
  placeholder_vals <- step_tbl[[var_col]]

  last_row <- window_raw_rows[nrow(window_raw_rows), , drop = FALSE]
  future_raw_rows <- last_row[rep(1L, length(steps)), , drop = FALSE]
  future_raw_rows[[raw_col]] <- placeholder_vals

  trial_data <- dplyr::bind_rows(train_data, window_raw_rows, future_raw_rows)
  target_idx_in_future <- which(steps == view$step)
  target_row_idx <- nrow(train_data) +
    nrow(window_raw_rows) +
    target_idx_in_future
  target_position <- nrow(train_data) + nrow(window_raw_rows) - (timesteps - 1)

  trial_vals <- seq(
    pred_val - bound,
    pred_val + bound,
    length.out = ctrl$trial_points
  )
  res <- purrr::map_dfr(
    trial_vals,
    kerasnip_trial_fit_step_view,
    trial_data = trial_data,
    view = view,
    level = level,
    raw_col = raw_col,
    target_row_idx = target_row_idx,
    target_position = target_position
  )
  kerasnip_compute_bound(res, pred_val)
}

#' Full Conformal Inference Method for `kerasnip_step_view` Objects
#'
#' @description
#' Full (refit-per-candidate) conformal intervals for one forecast step of a
#' multistep fit. Requires `step_lead()` and `step_sequence()` to share a
#' single source column (true of every multistep model built with this
#' package's own examples/vignette); see the design note above
#' `kerasnip_step_recipe_step()`. Only `control$method = "grid"` is
#' supported.
#'
#' @param object A `kerasnip_step_view`.
#' @param train_data The raw training data used to fit `object`'s underlying
#'   model.
#' @param ... Not used.
#' @param control A `probably::control_conformal_full()` object; defaults to
#'   `method = "grid"` if not supplied.
#' @return A `kerasnip_conformal_full_step`/`int_conformal_full` object;
#'   call `predict()` on it to get intervals for new data (a raw
#'   continuation of `train_data`).
#' @keywords internal
#' @exportS3Method probably::int_conformal_full
int_conformal_full.kerasnip_step_view <- function(
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
      "Only `\"grid\"` is supported for a multistep kerasnip step view.",
      i = paste0(
        "Pass `control = probably::control_conformal_full(",
        "method = \"grid\")`."
      )
    ))
  }

  seq_info <- kerasnip_step_recipe_step(object, "step_sequence")
  lead_info <- kerasnip_step_recipe_step(object, "step_lead")
  if (
    length(seq_info$columns) != 1 ||
      !identical(seq_info$columns, lead_info$columns)
  ) {
    rlang::abort(c(
      "int_conformal_full() for a step view requires `step_lead()` and",
      "`step_sequence()` to share a single source column.",
      i = paste0(
        "Found step_lead() column(s): ",
        paste(lead_info$columns, collapse = ", ")
      ),
      i = paste0(
        "and step_sequence() column(s): ",
        paste(seq_info$columns, collapse = ", ")
      )
    ))
  }

  var_mod <- kerasnip_step_var_model(object, train_data)
  object$.var_model <- var_mod
  structure(
    list(
      wflow = object,
      training = train_data,
      control = control,
      seq_info = seq_info,
      raw_col = seq_info$columns
    ),
    class = c("kerasnip_conformal_full_step", "int_conformal_full")
  )
}

#' Predict Method for `kerasnip_conformal_full_step` Objects
#'
#' @description
#' Computes full-conformal intervals for `new_data` (a raw continuation of
#' the training data), one grid search per surviving window via
#' `kerasnip_grid_one_step_view()`.
#'
#' @param object A `kerasnip_conformal_full_step` object, from
#'   `int_conformal_full.kerasnip_step_view()`.
#' @param new_data Raw data continuing the training series.
#' @param level The conformal level.
#' @param ... Not used.
#' @return A tibble with `.pred_lower`/`.pred_upper` columns, one row per
#'   window that survives `step_sequence()`'s history requirement.
#' @keywords internal
#' @exportS3Method stats::predict
predict.kerasnip_conformal_full_step <- function(
  object,
  new_data,
  level = 0.95,
  ...
) {
  rlang::check_dots_empty()
  view <- object$wflow
  new_pred <- kerasnip_setup_new_data(
    view,
    new_data,
    object$control$var_multiplier
  )
  full_pred <- predict(view$workflow, new_data = new_data)
  purrr::map_dfr(
    seq_len(nrow(new_pred)),
    kerasnip_grid_one_step_view,
    view = view,
    train_data = object$training,
    new_data = new_data,
    full_pred = full_pred,
    new_pred = new_pred,
    level = level,
    ctrl = object$control,
    seq_info = object$seq_info,
    raw_col = object$raw_col
  )
}
