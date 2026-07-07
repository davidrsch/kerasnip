#' Build a Sliding Window of Predictors for Sequence Models
#'
#' `step_sequence()` creates a *specification* of a recipe step that converts
#' one or more ordered numeric predictor columns into a single list-column of
#' `(timesteps, features)` matrices, one per row. This is the shape expected
#' by recurrent layer blocks (e.g. `keras3::layer_lstm()`,
#' `keras3::layer_gru()`) used with [create_keras_functional_spec()] or
#' [create_keras_sequential_spec()].
#'
#' @param recipe A recipe object. The step will be added to the sequence of
#'   operations for this recipe.
#' @param ... One or more selector functions to choose which (already
#'   time-ordered) numeric variables are windowed. See `[selections()]` for
#'   more details. All selected columns become "features" in the resulting
#'   window. For the `tidy` method, these are not currently used.
#' @param timesteps A single integer. The sliding window length (number of
#'   past rows, including the current one) to include in each window.
#' @param role For model terms created by this step, what analysis role
#'   should they be assigned?. By default, the new column is used as a
#'   predictor.
#' @param trained A logical to indicate if the quantities for preprocessing
#'   have been estimated.
#' @param columns A character string of the selected variable names. This is
#'   `NULL` until the step is trained by `[prep.recipe()]`.
#' @param new_col A character string for the name of the new list-column. The
#'   default is "sequence_matrix".
#' @param padding One of `"drop"` (default) or `"zero"`. Rows without a full
#'   `timesteps` history need special handling: `"drop"` removes them from
#'   the data (as `[recipes::step_naomit()]` does), while `"zero"` left-pads
#'   the missing history with rows of zeros so no rows are dropped.
#' @param skip A logical. Should the step be skipped when the recipe is
#'   baked by `[bake.recipe()]`? While all operations are baked when `prep`
#'   is run, skipping when `bake` is run may be other times when it is
#'   desirable to skip a processing step.
#' @param id A character string that is unique to this step to identify it.
#'
#' @return An updated version of `recipe` with the new step added to the
#'   sequence of existing steps (if any). For the `tidy` method, a tibble
#'   with columns `terms` (the selected column names), `value` (the name of
#'   the destination list-column), `timesteps`, and `id` (the step
#'   identifier).
#'
#' @examples
#' library(recipes)
#'
#' dat <- data.frame(x1 = 1:10, x2 = 11:20, y = 1:10)
#'
#' rec <- recipe(y ~ ., data = dat) %>%
#'   step_sequence(x1, x2, timesteps = 3, new_col = "window") %>%
#'   prep()
#'
#' bake(rec, new_data = NULL)
#' @importFrom recipes prep bake is_trained sel2char
#' @importFrom generics tidy
#' @importFrom tibble tibble
#' @export
step_sequence <- function(
  recipe,
  ...,
  timesteps,
  role = "predictor",
  trained = FALSE,
  columns = NULL,
  new_col = "sequence_matrix",
  padding = c("drop", "zero"),
  skip = FALSE,
  id = recipes::rand_id("sequence")
) {
  padding <- rlang::arg_match(padding)
  recipes::add_step(
    recipe,
    step_sequence_new(
      terms = enquos(...),
      role = role,
      trained = trained,
      columns = columns,
      timesteps = timesteps,
      new_col = new_col,
      padding = padding,
      skip = skip,
      id = id
    )
  )
}

step_sequence_new <- function(
  terms,
  role,
  trained,
  columns,
  timesteps,
  new_col,
  padding,
  skip,
  id
) {
  recipes::step(
    subclass = "sequence",
    terms = terms,
    role = role,
    trained = trained,
    columns = columns,
    timesteps = timesteps,
    new_col = new_col,
    padding = padding,
    skip = skip,
    id = id
  )
}

#' @export
prep.step_sequence <- function(x, training, info = NULL, ...) {
  col_names <- recipes::recipes_eval_select(x$terms, training, info)

  step_sequence_new(
    terms = x$terms,
    role = x$role,
    trained = TRUE,
    columns = col_names,
    timesteps = x$timesteps,
    new_col = x$new_col,
    padding = x$padding,
    skip = x$skip,
    id = x$id
  )
}

#' @export
bake.step_sequence <- function(object, new_data, ...) {
  if (object$skip) {
    return(new_data)
  }
  if (length(object$columns) == 0) {
    return(new_data)
  }
  recipes::check_new_data(object$columns, object, new_data)

  mat <- as.matrix(new_data[, object$columns, drop = FALSE])
  n <- nrow(mat)
  timesteps <- object$timesteps
  n_features <- ncol(mat)

  windows <- vector("list", n)
  keep <- rep(TRUE, n)

  for (i in seq_len(n)) {
    start <- i - timesteps + 1
    if (start < 1) {
      if (object$padding == "zero") {
        pad_n <- 1 - start
        windows[[i]] <- rbind(
          matrix(0, nrow = pad_n, ncol = n_features),
          mat[seq_len(i), , drop = FALSE]
        )
      } else {
        keep[i] <- FALSE
      }
    } else {
      windows[[i]] <- mat[start:i, , drop = FALSE]
    }
  }

  new_data[[object$new_col]] <- windows
  if (object$padding == "drop") {
    new_data <- new_data[keep, , drop = FALSE]
  }

  # drop original predictor columns
  new_data[, setdiff(names(new_data), object$columns), drop = FALSE]
}

#' @export
print.step_sequence <- function(x, ...) {
  if (is.null(x$columns) || length(x$columns) == 0) {
    cat("Sliding window of predictors (unprepped)\n")
  } else {
    cat(
      "Sliding window (timesteps =",
      x$timesteps,
      ") of predictors:",
      paste(x$columns, collapse = ", "),
      " -> ",
      x$new_col,
      "\n"
    )
  }
  invisible(x)
}

#' @importFrom generics required_pkgs
#' @export
required_pkgs.step_sequence <- function(x, ...) {
  c("kerasnip")
}

#' @export
tidy.step_sequence <- function(x, ...) {
  if (recipes::is_trained(x)) {
    if (length(x$columns) > 0) {
      tibble::tibble(
        terms = x$columns,
        value = x$new_col,
        timesteps = x$timesteps,
        id = x$id
      )
    } else {
      tibble::tibble(
        terms = character(),
        value = character(),
        timesteps = integer(),
        id = character()
      )
    }
  } else {
    tibble::tibble(
      terms = recipes::sel2char(x$terms),
      value = NA_character_,
      timesteps = x$timesteps,
      id = x$id
    )
  }
}
