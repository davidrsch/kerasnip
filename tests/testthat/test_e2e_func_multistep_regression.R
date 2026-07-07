test_that("E2E: LSTM multi-step-ahead regression works end-to-end", {
  skip_if_no_keras()
  options(kerasnip.show_removal_messages = FALSE)
  on.exit(options(kerasnip.show_removal_messages = TRUE), add = TRUE)

  set.seed(42)
  n <- 120
  timesteps <- 8
  horizon <- 3
  dat <- tibble::tibble(value = sin(seq_len(n) / 10) + rnorm(n, sd = 0.05))

  # step_lead() must run before step_sequence() when both draw on the same
  # raw column: step_sequence() consumes (and drops) its source column, so
  # step_lead() needs to see it first. step_naomit() defaults to skip = TRUE,
  # so it only drops rows at training time, not when baking new data at
  # predict time (where future values are legitimately unknown/NA).
  rec <- recipe(dat) |>
    step_lead(value, lead = seq_len(horizon), prefix = "lead_") |>
    step_naomit(starts_with("lead_")) |>
    step_sequence(value, timesteps = timesteps, new_col = "window")

  input_block <- function(input_shape) {
    keras3::layer_input(shape = input_shape, name = "window_input")
  }
  lstm_block <- function(tensor, units = 8) {
    tensor |> keras3::layer_lstm(units = units)
  }
  # `units` needs a default to work around a doc-generator quirk when
  # handling args with no default (see test_e2e_func_classification.R).
  output_block <- function(tensor, units = 1) {
    tensor |> keras3::layer_dense(units = units)
  }

  model_name <- "e2e_multistep_reg"
  on.exit(suppressMessages(remove_keras_spec(model_name)), add = TRUE)

  create_keras_functional_spec(
    model_name = model_name,
    layer_blocks = list(
      window = input_block,
      lstm = inp_spec(lstm_block, "window"),
      output = inp_spec(output_block, "lstm")
    ),
    mode = "regression"
  )

  spec <- e2e_multistep_reg(
    output_units = horizon,
    fit_epochs = 2,
    fit_verbose = 0
  ) |>
    set_engine("keras")

  wf <- workflows::workflow(rec, spec)

  expect_no_error(fit_obj <- parsnip::fit(wf, data = dat))

  new_data <- dat[seq_len(timesteps + 2), , drop = FALSE]
  preds <- predict(fit_obj, new_data = new_data)

  expect_s3_class(preds, "tbl_df")
  expect_equal(names(preds), ".pred")
  expect_true(is.list(preds$.pred))
  expect_equal(nrow(preds), 3) # (timesteps + 2) - (timesteps - 1) dropped rows

  # Each row's nested tibble has one row per forecast step, correctly numbered
  first_step_tbl <- preds$.pred[[1]]
  expect_s3_class(first_step_tbl, "tbl_df")
  expect_named(first_step_tbl, c(".step", ".pred"))
  expect_equal(first_step_tbl$.step, seq_len(horizon))
  expect_true(is.numeric(first_step_tbl$.pred))
  expect_equal(nrow(first_step_tbl), horizon)

  # conf_int/pred_int: each forecast step gets its own independent Laplace
  # posterior (own tau/sigma_sq_noise), rather than one width pooled across
  # all steps. With a minimally-trained model (fit_epochs = 2) interval
  # widths need not grow monotonically with the horizon on every run, so
  # this only checks the intervals are well-formed, not a specific trend --
  # see the multistep_forecasting vignette for a worked example showing
  # widening intervals under realistic training.
  preds_ci <- predict(fit_obj, new_data = new_data, type = "conf_int")
  first_ci_tbl <- preds_ci$.pred[[1]]
  expect_named(first_ci_tbl, c(".step", ".pred", ".pred_lower", ".pred_upper"))
  expect_equal(nrow(first_ci_tbl), horizon)
  expect_true(all(first_ci_tbl$.pred_lower <= first_ci_tbl$.pred))
  expect_true(all(first_ci_tbl$.pred <= first_ci_tbl$.pred_upper))
  ci_width <- first_ci_tbl$.pred_upper - first_ci_tbl$.pred_lower

  preds_pi <- predict(fit_obj, new_data = new_data, type = "pred_int")
  first_pi_tbl <- preds_pi$.pred[[1]]
  expect_named(first_pi_tbl, c(".step", ".pred", ".pred_lower", ".pred_upper"))
  # Prediction intervals (epistemic + aleatoric noise) must be at least as
  # wide as confidence intervals (epistemic only) at every step.
  pi_width <- first_pi_tbl$.pred_upper - first_pi_tbl$.pred_lower
  expect_true(all(pi_width >= ci_width))

  # joint = TRUE: correlated sample trajectories across steps (raw draws, not
  # a pre-summarized interval), tagged with the tidybayes-style `.draw`
  # convention. Only supported for pred_int; conf_int has no modeled
  # source of cross-step correlation in this implementation.
  n_draws <- 300L
  preds_joint <- predict(
    fit_obj,
    new_data = new_data,
    type = "pred_int",
    joint = TRUE,
    n_draws = n_draws
  )
  expect_s3_class(preds_joint, "tbl_df")
  expect_equal(names(preds_joint), ".pred")
  expect_equal(nrow(preds_joint), 3)

  first_joint_tbl <- preds_joint$.pred[[1]]
  expect_named(first_joint_tbl, c(".draw", ".step", ".pred"))
  expect_equal(nrow(first_joint_tbl), n_draws * horizon)
  expect_equal(sort(unique(first_joint_tbl$.draw)), seq_len(n_draws))
  expect_equal(sort(unique(first_joint_tbl$.step)), seq_len(horizon))

  # The whole point of `joint = TRUE`: draws at different steps should be
  # correlated (not independent), since they share the jointly-estimated
  # noise covariance from training residuals.
  wide_joint <- tidyr::pivot_wider(
    first_joint_tbl,
    names_from = .step,
    values_from = .pred,
    names_prefix = "step_"
  )
  step_cor <- stats::cor(wide_joint$step_1, wide_joint$step_2)
  expect_true(abs(step_cor) > 0.1)

  expect_error(
    predict(fit_obj, new_data = new_data, type = "conf_int", joint = TRUE),
    "only supported for.*pred_int"
  )
  # joint = TRUE must not be silently ignored for the default (numeric) type
  expect_error(
    predict(fit_obj, new_data = new_data, joint = TRUE),
    "only supported for.*pred_int"
  )
})
