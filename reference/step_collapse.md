# Collapse Predictors into a single list-column

`step_collapse()` creates a a *specification* of a recipe step that will
convert a group of predictors into a single list-column. This is useful
for custom models that need the predictors in a different format.

## Usage

``` r
step_collapse(
  recipe,
  ...,
  role = "predictor",
  trained = FALSE,
  columns = NULL,
  new_col = "predictor_matrix",
  skip = FALSE,
  id = recipes::rand_id("collapse")
)
```

## Arguments

- recipe:

  A recipe object. The step will be added to the sequence of operations
  for this recipe.

- ...:

  One or more selector functions to choose which variables are affected
  by the step. See `[selections()]` for more details. For the `tidy`
  method, these are not currently used.

- role:

  For model terms created by this step, what analysis role should they
  be assigned?. By default, the new columns are used as predictors.

- trained:

  A logical to indicate if the quantities for preprocessing have been
  estimated.

- columns:

  A character string of the selected variable names. This is `NULL`
  until the step is trained by `[prep.recipe()]`.

- new_col:

  A character string for the name of the new list-column. The default is
  "predictor_matrix".

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
`terms` which is the columns that are affected and `value` which is the
type of collapse.

## Examples

``` r
library(recipes)
#> Loading required package: dplyr
#> 
#> Attaching package: ‘dplyr’
#> The following objects are masked from ‘package:stats’:
#> 
#>     filter, lag
#> The following objects are masked from ‘package:base’:
#> 
#>     intersect, setdiff, setequal, union
#> 
#> Attaching package: ‘recipes’
#> The following object is masked from ‘package:stats’:
#> 
#>     step

# 2 predictors
dat <- data.frame(
  x1 = 1:10,
  x2 = 11:20,
  y = 1:10
)

rec <- recipe(y ~ ., data = dat) %>%
  step_collapse(x1, x2, new_col = "pred") %>%
  prep()

bake(rec, new_data = NULL)
#> # A tibble: 10 × 2
#>        y pred         
#>    <int> <list>       
#>  1     1 <int [1 × 2]>
#>  2     2 <int [1 × 2]>
#>  3     3 <int [1 × 2]>
#>  4     4 <int [1 × 2]>
#>  5     5 <int [1 × 2]>
#>  6     6 <int [1 × 2]>
#>  7     7 <int [1 × 2]>
#>  8     8 <int [1 × 2]>
#>  9     9 <int [1 × 2]>
#> 10    10 <int [1 × 2]>
```
