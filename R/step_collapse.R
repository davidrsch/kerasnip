#' Collapse Predictors into a single list-column
#'
#' `step_collapse()` creates a a *specification* of a recipe step that will
#'  convert a group of predictors into a single list-column. This is useful
#'  for custom models that need the predictors in a different format.
#'
#' @param recipe A recipe object. The step will be added to the sequence of
#'   operations for this recipe.
#' @param ... One or more selector functions to choose which variables are
#'   affected by the step. See `[selections()]` for more details. For the `tidy`
#'   method, these are not currently used.
#' @param role For model terms created by this step, what analysis role should
#'   they be assigned?. By default, the new columns are used as predictors.
#' @param trained A logical to indicate if the quantities for preprocessing
#'   have been estimated.
#' @param columns A character string of the selected variable names. This is
#'   `NULL` until the step is trained by `[prep.recipe()]`.
#' @param new_col A character string for the name of the new list-column. The
#'   default is "predictor_matrix".
#' @param skip A logical. Should the step be skipped when the recipe is
#'   baked by `[bake.recipe()]`? While all operations are baked when `prep` is run,
#'   skipping when `bake` is run may be other times when it is desirable to
#'   skip a processing step.
#' @param id A character string that is unique to this step to identify it.
#'
#' @return An updated version of `recipe` with the new step added to the
#'   sequence of existing steps (if any). For the `tidy` method, a tibble with
#'   columns `terms` which is the columns that are affected and `value` which is
#'   the type of collapse.
#'
#' @examples
#' library(recipes)
#'
#' # 2 predictors
#' dat <- data.frame(
#'   x1 = 1:10,
#'   x2 = 11:20,
#'   y = 1:10
#' )
#'
#' rec <- recipe(y ~ ., data = dat) %>%
#'   step_collapse(x1, x2, new_col = "pred") %>%
#'   prep()
#'
#' bake(rec, new_data = NULL)
#' @export
step_collapse <- function(
  recipe,
  ...,
  role = "predictor",
  trained = FALSE,
  columns = NULL,
  new_col = "predictor_matrix",
  skip = FALSE,
  id = recipes::rand_id("collapse")
) {
  recipes::add_step(
    recipe,
    step_collapse_new(
      terms = enquos(...),
      role = role,
      trained = trained,
      columns = columns,
      new_col = new_col,
      skip = skip,
      id = id
    )
  )
}

step_collapse_new <- function(
  terms,
  role,
  trained,
  columns,
  new_col,
  skip,
  id
) {
  recipes::step(
    subclass = "collapse",
    terms = terms,
    role = role,
    trained = trained,
    columns = columns,
    new_col = new_col,
    skip = skip,
    id = id
  )
}

#' @export
prep.step_collapse <- function(x, training, info = NULL, ...) {
  col_names <- recipes::recipes_eval_select(x$terms, training, info)

  step_collapse_new(
    terms = x$terms,
    role = x$role,
    trained = TRUE,
    columns = col_names,
    new_col = x$new_col,
    skip = x$skip,
    id = x$id
  )
}

#' @export
bake.step_collapse <- function(object, new_data, ...) {
  recipes::check_new_data(object$columns, object, new_data)

  rows_list <- apply(
    new_data[, object$columns, drop = FALSE],
    1,
    function(row) matrix(row, nrow = 1),
    simplify = FALSE
  )

  new_data[[object$new_col]] <- rows_list

  # drop original predictor columns
  new_data <- new_data[, setdiff(names(new_data), object$columns), drop = FALSE]

  new_data
}

#' @export
print.step_collapse <- function(x, ...) {
  if (is.null(x$columns)) {
    cat("Collapse predictors into list-column (unprepped)\n")
  } else {
    cat(
      "Collapse predictors into list-column:",
      paste(x$columns, collapse = ", "),
      " → ",
      x$new_col,
      "\n"
    )
  }
  invisible(x)
}
