dat <- data.frame(
  x1 = 1:10,
  y = 1:10
)

test_that("step_lead works with a single lead value", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1)

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_true("lead_1_y" %in% names(baked_dat))
  expect_true("y" %in% names(baked_dat)) # keep_original_cols = TRUE by default
  expect_equal(baked_dat$lead_1_y, c(2:10, NA))
})

test_that("step_lead works with a vector of lead values", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1:3)

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_true(all(c("lead_1_y", "lead_2_y", "lead_3_y") %in% names(baked_dat)))
  expect_equal(baked_dat$lead_1_y, c(2:10, NA))
  expect_equal(baked_dat$lead_2_y, c(3:10, NA, NA))
  expect_equal(baked_dat$lead_3_y, c(4:10, NA, NA, NA))
})

test_that("default argument fills trailing rows", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1, default = 0)

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_equal(baked_dat$lead_1_y, c(2:10, 0))
})

test_that("prefix argument works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1, prefix = "future_")

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_true("future_1_y" %in% names(baked_dat))
})

test_that("keep_original_cols = FALSE drops the source column", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1, keep_original_cols = FALSE)

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_false("y" %in% names(baked_dat))
  expect_true("lead_1_y" %in% names(baked_dat))
})

test_that("step_lead composes with step_naomit to drop incomplete rows", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1:2) |>
    step_naomit(starts_with("lead_"))

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  # Last 2 rows lack a full 2-step-ahead future window
  expect_equal(nrow(baked_dat), 8)
})

test_that("skip argument works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1, skip = TRUE)

  prepped_rec <- prep(rec)
  baked_dat <- bake(prepped_rec, new_data = NULL)

  expect_false("lead_1_y" %in% names(baked_dat))
})

test_that("prep validates lead is a nonnegative integer", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = -1)

  expect_error(prep(rec))
})

test_that("print method works", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1:2)

  expect_snapshot(print(rec$steps[[1]]))

  prepped_rec <- prep(rec)
  expect_snapshot(print(prepped_rec$steps[[1]]))
})

test_that("step_lead handles selectors that don't match", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(non_existent_col, lead = 1)

  expect_error(prep(rec))
})

test_that("required_pkgs.step_lead returns kerasnip", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1)
  expect_equal(recipes::required_pkgs(rec$steps[[1]]), "kerasnip")
})

test_that("tidy.step_lead works before and after prep", {
  rec <- recipe(y ~ ., data = dat) |>
    step_lead(y, lead = 1:2)

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
  expect_equal(unname(prepped_tidy$terms), c("y", "y"))
  expect_setequal(prepped_tidy$value, c("lead_1_y", "lead_2_y"))
})

test_that("tidy.step_lead returns empty tibble when trained with no columns", {
  empty_step <- kerasnip:::step_lead_new(
    terms = rlang::quos(),
    role = "outcome",
    trained = TRUE,
    lead = 1,
    prefix = "lead_",
    default = NA,
    columns = character(0),
    keep_original_cols = TRUE,
    skip = FALSE,
    id = "lead_empty"
  )
  result <- tidy(empty_step)
  expect_s3_class(result, "tbl_df")
  expect_equal(nrow(result), 0L)
  expect_named(result, c("terms", "value", "id"))
})
