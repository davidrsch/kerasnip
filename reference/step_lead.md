# Create a Lead Predictor

`step_lead()` creates a *specification* of a recipe step that will
create one or more new columns of data that are leading (i.e. future)
values of existing columns. This is the target-side companion to
`[recipes::step_lag()]` (which only supports positive lag/past values,
not lead/future values) and is intended for building multi-step-ahead
forecasting targets, e.g. `step_lead(y, lead = 1:6)` produces the next
six values of `y` as separate columns, one per row.

## Usage

``` r
step_lead(
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
)
```

## Arguments

- recipe:

  A recipe object. The step will be added to the sequence of operations
  for this recipe.

- ...:

  One or more selector functions to choose which variables are leading.
  See `[selections()]` for more details. For the `tidy` method, these
  are not currently used.

- lead:

  A vector of nonnegative integers. Each value produces a leading column
  for each selected variable.

- prefix:

  A prefix added to the leading columns names. The default naming
  convention is `<prefix><lead value>_<original variable name>`, e.g.
  `lead_1_value`.

- default:

  Value to fill in the trailing rows that don't have a complete future
  window (analogous to `default` in `[recipes::step_lag()]`). Defaults
  to `NA`.

- role:

  For model terms created by this step, what analysis role should they
  be assigned?. By default, the new columns are used as outcomes.

- trained:

  A logical to indicate if the quantities for preprocessing have been
  estimated.

- columns:

  A character string of the selected variable names. This is `NULL`
  until the step is trained by `[prep.recipe()]`.

- keep_original_cols:

  A logical to keep the original variables in the output. Defaults to
  `TRUE`.

- skip:

  A logical. Should the step be skipped when the recipe is baked by
  `[bake.recipe()]`? While all operations are baked when `prep` is run,
  skipping when `bake` is run may be other times when it is desirable to
  skip a processing step.

- id:

  A character string that is unique to this step to identify it.

## Value

An updated version of `recipe` with the new step added to the sequence
of existing steps (if any). For the `tidy` method, a tibble with columns
`terms` (the selected column names), `value` (the name of the resulting
leading column), and `id` (the step identifier).

## Details

Combine with `[recipes::step_naomit()]` (e.g.
`step_naomit(starts_with(prefix))`) to drop the trailing rows that don't
have a full future window, mirroring how `[step_sequence()]`'s
`padding = "drop"` removes rows lacking a full past window.

## Examples

``` r
library(recipes)

dat <- data.frame(y = 1:10)

rec <- recipe(y ~ ., data = dat) %>%
  step_lead(y, lead = 1:2) %>%
  prep()

bake(rec, new_data = NULL)
#> # A tibble: 10 × 3
#>        y lead_1_y lead_2_y
#>    <int>    <int>    <int>
#>  1     1        2        3
#>  2     2        3        4
#>  3     3        4        5
#>  4     4        5        6
#>  5     5        6        7
#>  6     6        7        8
#>  7     7        8        9
#>  8     8        9       10
#>  9     9       10       NA
#> 10    10       NA       NA
```
