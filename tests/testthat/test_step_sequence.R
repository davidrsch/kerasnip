dat <- data.frame(
  x1 = 1:10,
  x2 = 11:20,
  y = 1:10
)

test_that("step_sequence works with basic selectors (padding = drop)", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(x1, timesteps = 3, padding = "drop")

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  # First 2 rows dropped (not enough history for timesteps = 3)
  expect_equal(nrow(baked_dat), 8)
  expect_true("sequence_matrix" %in% names(baked_dat))
  expect_false("x1" %in% names(baked_dat))
  expect_true(is.list(baked_dat$sequence_matrix))
  expect_equal(
    vapply(baked_dat$sequence_matrix, nrow, numeric(1)),
    rep(3, 8)
  )
  expect_equal(
    vapply(baked_dat$sequence_matrix, ncol, numeric(1)),
    rep(1, 8)
  )
  # The last window should be the last 3 raw values, in order
  expect_equal(as.vector(baked_dat$sequence_matrix[[8]]), 8:10)
})

test_that("step_sequence works with multiple feature columns", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(x1, x2, timesteps = 2, new_col = "window")

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_equal(nrow(baked_dat), 9)
  expect_true("window" %in% names(baked_dat))
  expect_equal(
    vapply(baked_dat$window, ncol, numeric(1)),
    rep(2, 9)
  )
  expect_equal(baked_dat$window[[9]], cbind(x1 = 9:10, x2 = 19:20))
})

test_that("padding = 'zero' keeps all rows and left-pads with zeros", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(x1, timesteps = 3, padding = "zero")

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_equal(nrow(baked_dat), 10)
  # First row: only 1 real value, padded with 2 zero rows
  expect_equal(as.vector(baked_dat$sequence_matrix[[1]]), c(0, 0, 1))
  # Second row: 2 real values, padded with 1 zero row
  expect_equal(as.vector(baked_dat$sequence_matrix[[2]]), c(0, 1, 2))
  # Fully-populated window unaffected by padding
  expect_equal(as.vector(baked_dat$sequence_matrix[[10]]), 8:10)
})

test_that("new_col argument works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(x1, timesteps = 2, new_col = "seq_col")

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_true("seq_col" %in% names(baked_dat))
})

test_that("skip argument works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(x1, timesteps = 2, skip = TRUE)

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_equal(ncol(baked_dat), 3)
  expect_false("sequence_matrix" %in% names(baked_dat))
})

test_that("print method works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(x1, timesteps = 3)

  expect_snapshot(print(rec$steps[[1]]))

  prepped_rec <- prep(rec)
  expect_snapshot(print(prepped_rec$steps[[1]]))
})

test_that("step_sequence handles selectors that don't match", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(non_existent_col, timesteps = 3)

  expect_error(prep(rec))
})

test_that("required_pkgs.step_sequence returns kerasnip", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(x1, timesteps = 3)
  expect_equal(recipes::required_pkgs(rec$steps[[1]]), "kerasnip")
})

test_that("tidy.step_sequence works before and after prep", {
  rec <- recipe(y ~ ., data = dat) |>
    step_sequence(x1, timesteps = 3, new_col = "window")

  # Before prep: terms from selector, NA value
  unprepped_tidy <- tidy(rec, number = 1)
  expect_s3_class(unprepped_tidy, "tbl_df")
  expect_named(unprepped_tidy, c("terms", "value", "timesteps", "id"))
  expect_true(all(is.na(unprepped_tidy$value)))
  expect_equal(unprepped_tidy$timesteps, 3)

  # After prep: actual column names and destination
  prepped_rec <- prep(rec)
  prepped_tidy <- tidy(prepped_rec, number = 1)
  expect_s3_class(prepped_tidy, "tbl_df")
  expect_named(prepped_tidy, c("terms", "value", "timesteps", "id"))
  expect_equal(unname(prepped_tidy$terms), "x1")
  expect_true(all(prepped_tidy$value == "window"))
  expect_equal(prepped_tidy$timesteps, 3)
})

test_that("tidy.step_sequence: empty tibble when trained with no columns", {
  empty_step <- kerasnip:::step_sequence_new(
    terms = rlang::quos(),
    role = "predictor",
    trained = TRUE,
    columns = character(0),
    timesteps = 3,
    new_col = "sequence_matrix",
    padding = "drop",
    skip = FALSE,
    id = "sequence_empty"
  )
  result <- tidy(empty_step)
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0L)
  expect_named(result, c("terms", "value", "timesteps", "id"))
})
