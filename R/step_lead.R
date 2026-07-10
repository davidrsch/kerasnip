#' Create a Lead Predictor
#'
#' `step_lead()` creates a *specification* of a recipe step that will create
#' one or more new columns of data that are leading (i.e. future) values of
#' existing columns. This is the target-side companion to
#' `[recipes::step_lag()]` (which only supports positive lag/past values, not
#' lead/future values) and is intended for building multi-step-ahead
#' forecasting targets, e.g. `step_lead(y, lead = 1:6)` produces the next six
#' values of `y` as separate columns, one per row.
#'
#' @param recipe A recipe object. The step will be added to the sequence of
#'   operations for this recipe.
#' @param ... One or more selector functions to choose which variables are
#'   leading. See `[selections()]` for more details. For the `tidy` method,
#'   these are not currently used.
#' @param lead A vector of nonnegative integers. Each value produces a
#'   leading column for each selected variable.
#' @param prefix A prefix added to the leading columns names. The default
#'   naming convention is `<prefix><lead value>_<original variable name>`,
#'   e.g. `lead_1_value`.
#' @param default Value to fill in the trailing rows that don't have a
#'   complete future window (analogous to `default` in
#'   `[recipes::step_lag()]`). Defaults to `NA`.
#' @param role For model terms created by this step, what analysis role
#'   should they be assigned?. By default, the new columns are used as
#'   outcomes.
#' @param trained A logical to indicate if the quantities for preprocessing
#'   have been estimated.
#' @param columns A character string of the selected variable names. This is
#'   `NULL` until the step is trained by `[prep.recipe()]`.
#' @param keep_original_cols A logical to keep the original variables in the
#'   output. Defaults to `TRUE`.
#' @param skip A logical. Should the step be skipped when the recipe is
#'   baked by `[bake.recipe()]`? While all operations are baked when `prep`
#'   is run, skipping when `bake` is run may be other times when it is
#'   desirable to skip a processing step.
#' @param id A character string that is unique to this step to identify it.
#'
#' @return An updated version of `recipe` with the new step added to the
#'   sequence of existing steps (if any). For the `tidy` method, a tibble
#'   with columns `terms` (the selected column names), `value` (the name of
#'   the resulting leading column), and `id` (the step identifier).
#'
#' @details
#' Combine with `[recipes::step_naomit()]` (e.g.
#' `step_naomit(starts_with(prefix))`) to drop the trailing rows that don't
#' have a full future window, mirroring how `[step_sequence()]`'s
#' `padding = "drop"` removes rows lacking a full past window.
#'
#' @examples
#' library(recipes)
#'
#' dat <- data.frame(y = 1:10)
#'
#' rec <- recipe(y ~ ., data = dat) %>%
#'   step_lead(y, lead = 1:2) %>%
#'   prep()
#'
#' bake(rec, new_data = NULL)
#' @importFrom recipes prep bake is_trained sel2char
#' @importFrom generics tidy
#' @importFrom tibble tibble
#' @export
step_lead <- function(
  recipe,
  ...,
  lead = 1,
  prefix = "lead_",
  default = NA,
  role = "outcome",
  trained = FALSE,
  columns = NULL,
  keep_original_cols = TRUE,
  skip = FALSE,
  id = recipes::rand_id("lead")
) {
  recipes::add_step(
    recipe,
    step_lead_new(
      terms = enquos(...),
      role = role,
      trained = trained,
      lead = lead,
      prefix = prefix,
      default = default,
      columns = columns,
      keep_original_cols = keep_original_cols,
      skip = skip,
      id = id
    )
  )
}

step_lead_new <- function(
  terms,
  role,
  trained,
  lead,
  prefix,
  default,
  columns,
  keep_original_cols,
  skip,
  id
) {
  recipes::step(
    subclass = "lead",
    terms = terms,
    role = role,
    trained = trained,
    lead = lead,
    prefix = prefix,
    default = default,
    columns = columns,
    keep_original_cols = keep_original_cols,
    skip = skip,
    id = id
  )
}

#' @export
prep.step_lead <- function(x, training, info = NULL, ...) {
  if (!all(x$lead == as.integer(x$lead)) || any(x$lead < 0)) {
    rlang::abort("`lead` argument must be nonnegative integer-valued.")
  }
  col_names <- recipes::recipes_eval_select(x$terms, training, info)

  step_lead_new(
    terms = x$terms,
    role = x$role,
    trained = TRUE,
    lead = x$lead,
    prefix = x$prefix,
    default = x$default,
    columns = col_names,
    keep_original_cols = x$keep_original_cols,
    skip = x$skip,
    id = x$id
  )
}

#' @export
bake.step_lead <- function(object, new_data, ...) {
  if (object$skip) {
    return(new_data)
  }
  col_names <- object$columns
  if (length(col_names) == 0) {
    return(new_data)
  }
  recipes::check_new_data(col_names, object, new_data)

  for (col_name in col_names) {
    for (lead_val in object$lead) {
      new_name <- paste0(object$prefix, lead_val, "_", col_name)
      new_data[[new_name]] <- dplyr::lead(
        new_data[[col_name]],
        lead_val,
        default = object$default
      )
    }
  }

  if (!isTRUE(object$keep_original_cols)) {
    new_data <- new_data[, setdiff(names(new_data), col_names), drop = FALSE]
  }

  new_data
}

#' @export
print.step_lead <- function(x, ...) {
  if (is.null(x$columns) || length(x$columns) == 0) {
    cat("Leading predictors (unprepped)\n")
  } else {
    cat(
      "Leading (",
      paste(x$lead, collapse = ", "),
      ") predictors:",
      paste(x$columns, collapse = ", "),
      "\n"
    )
  }
  invisible(x)
}

#' @importFrom generics required_pkgs
#' @export
required_pkgs.step_lead <- function(x, ...) {
  c("kerasnip")
}

#' @export
tidy.step_lead <- function(x, ...) {
  if (recipes::is_trained(x)) {
    if (length(x$columns) > 0) {
      tibble::tibble(
        terms = rep(x$columns, each = length(x$lead)),
        value = paste0(
          x$prefix,
          rep(x$lead, times = length(x$columns)),
          "_",
          rep(x$columns, each = length(x$lead))
        ),
        id = x$id
      )
    } else {
      tibble::tibble(
        terms = character(),
        value = character(),
        id = character()
      )
    }
  } else {
    tibble::tibble(
      terms = recipes::sel2char(x$terms),
      value = NA_character_,
      id = x$id
    )
  }
}
