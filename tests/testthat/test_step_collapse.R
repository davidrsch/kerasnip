dat <- data.frame(
  x1 = 1:10,
  x2 = 11:20,
  x3 = 21:30,
  y = 1:10
)

test_that("step_collapse works with basic selectors", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(x1, x2)

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_equal(ncol(baked_dat), 3)
  expect_true("predictor_matrix" %in% names(baked_dat))
  expect_true(is.list(baked_dat$predictor_matrix))
  expect_equal(length(baked_dat$predictor_matrix), 10)
  expect_equal(vapply(baked_dat$predictor_matrix, nrow, numeric(1)), rep(1, 10))
  expect_equal(vapply(baked_dat$predictor_matrix, ncol, numeric(1)), rep(2, 10))
})

test_that("step_collapse works with tidyselect helpers", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(starts_with("x"))

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_equal(ncol(baked_dat), 2)
  expect_true("predictor_matrix" %in% names(baked_dat))
  expect_true(is.list(baked_dat$predictor_matrix))
  expect_equal(length(baked_dat$predictor_matrix), 10)
  expect_equal(vapply(baked_dat$predictor_matrix, nrow, numeric(1)), rep(1, 10))
  expect_equal(vapply(baked_dat$predictor_matrix, ncol, numeric(1)), rep(3, 10))
})

test_that("new_col argument works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(x1, x2, new_col = "collapsed_predictors")

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_true("collapsed_predictors" %in% names(baked_dat))
})

test_that("skip argument works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(x1, x2, skip = TRUE)

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_equal(ncol(baked_dat), 4)
  expect_false("predictor_matrix" %in% names(baked_dat))
})

test_that("print method works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(x1, x2)

  expect_snapshot(print(rec$steps[[1]]))

  prepped_rec <- prep(rec)
  expect_snapshot(print(prepped_rec$steps[[1]]))
})

test_that("print method works for prepped recipe", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(x1, x2) |>
    prep()

  expect_snapshot(print(rec$steps[[1]]))
})

test_that("step_collapse handles no selectors", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse()

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_equal(ncol(baked_dat), 4)
  expect_false("predictor_matrix" %in% names(baked_dat))
})

test_that("step_collapse handles selectors that don't match", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(non_existent_col)

  expect_error(prep(rec))
})

test_that("required_pkgs.step_collapse returns kerasnip", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(x1, x2)
  expect_equal(recipes::required_pkgs(rec), "kerasnip")
})

test_that("tidy.step_collapse works before and after prep", {
  rec <- recipe(y ~ ., data = dat) |>
    step_collapse(x1, x2, new_col = "pred")

  # Before prep: terms from selector, NA value
  unprepped_tidy <- tidy(rec, number = 1)
  expect_s3_class(unprepped_tidy, "tbl_df")
  expect_named(unprepped_tidy, c("terms", "value", "id"))
  expect_true(all(is.na(unprepped_tidy$value)))

  # After prep: actual column names and destination
  prepped_rec <- prep(rec)
  prepped_tidy <- tidy(prepped_rec, number = 1)
  expect_s3_class(prepped_tidy, "tbl_df")
  expect_named(prepped_tidy, c("terms", "value", "id"))
  expect_equal(sort(prepped_tidy$terms), c("x1", "x2"))
  expect_true(all(prepped_tidy$value == "pred"))
})
