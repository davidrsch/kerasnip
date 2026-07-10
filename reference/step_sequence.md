# Build a Sliding Window of Predictors for Sequence Models

`step_sequence()` creates a *specification* of a recipe step that
converts one or more ordered numeric predictor columns into a single
list-column of `(timesteps, features)` matrices, one per row. This is
the shape expected by recurrent layer blocks (e.g.
[`keras3::layer_lstm()`](https://keras3.posit.co/reference/layer_lstm.html),
[`keras3::layer_gru()`](https://keras3.posit.co/reference/layer_gru.html))
used with
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_functional_spec.md)
or
[`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_sequential_spec.md).

## Usage

``` r
step_sequence(
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
)
```

## Arguments

- recipe:

  A recipe object. The step will be added to the sequence of operations
  for this recipe.

- ...:

  One or more selector functions to choose which (already time-ordered)
  numeric variables are windowed. See `[selections()]` for more details.
  All selected columns become "features" in the resulting window. For
  the `tidy` method, these are not currently used.

- timesteps:

  A single integer. The sliding window length (number of past rows,
  including the current one) to include in each window.

- role:

  For model terms created by this step, what analysis role should they
  be assigned?. By default, the new column is used as a predictor.

- trained:

  A logical to indicate if the quantities for preprocessing have been
  estimated.

- columns:

  A character string of the selected variable names. This is `NULL`
  until the step is trained by `[prep.recipe()]`.

- new_col:

  A character string for the name of the new list-column. The default is
  "sequence_matrix".

- padding:

  One of `"drop"` (default) or `"zero"`. Rows without a full `timesteps`
  history need special handling: `"drop"` removes them from the data (as
  `[recipes::step_naomit()]` does), while `"zero"` left-pads the missing
  history with rows of zeros so no rows are dropped.

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
`terms` (the selected column names), `value` (the name of the
destination list-column), `timesteps`, and `id` (the step identifier).

## Examples

``` r
library(recipes)

dat <- data.frame(x1 = 1:10, x2 = 11:20, y = 1:10)

rec <- recipe(y ~ ., data = dat) %>%
  step_sequence(x1, x2, timesteps = 3, new_col = "window") %>%
  prep()

bake(rec, new_data = NULL)
#> # A tibble: 8 × 2
#>       y window       
#>   <int> <list>       
#> 1     3 <int [3 × 2]>
#> 2     4 <int [3 × 2]>
#> 3     5 <int [3 × 2]>
#> 4     6 <int [3 × 2]>
#> 5     7 <int [3 × 2]>
#> 6     8 <int [3 × 2]>
#> 7     9 <int [3 × 2]>
#> 8    10 <int [3 × 2]>
```
