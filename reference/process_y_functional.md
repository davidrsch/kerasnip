# Process Outcome Input for Keras (Functional API)

Preprocesses outcome data (`y`) into a format suitable for Keras models
built with the Functional API. Handles both regression (numeric) and
classification (factor) outcomes, including one-hot encoding for
classification, and supports multiple outputs.

## Usage

``` r
process_y_functional(
  y,
  is_classification = NULL,
  class_levels = NULL,
  layer_blocks = NULL
)
```

## Arguments

- y:

  A vector or data frame of outcomes.

- is_classification:

  Logical, optional. If `TRUE`, treats `y` as classification. If
  `FALSE`, treats as regression. If `NULL` (default), it's determined
  from `is.factor(y)`.

- class_levels:

  Character vector, optional. The factor levels for classification
  outcomes. If `NULL` (default), determined from `levels(y)`.

- layer_blocks:

  A named list of layer block functions, optional. Used to disambiguate
  a multi-column `y` between the "N independent named output heads" case
  (one block per column name, e.g. `output_1`, `output_2`) and the
  "single vector-valued output" case (e.g. multi-step regression, a
  single block named `"output"` with `units = ncol(y)`). If `NULL`
  (default, and for any caller that predates this parameter), the
  original per-column-split behavior is preserved.

## Value

A list containing:

- `y_proc`: The processed outcome data (matrix or one-hot encoded array,
  or list of these for multiple outputs).

- `is_classification`: Logical, indicating if `y` was treated as
  classification.

- `num_classes`: Integer, the number of classes for classification, or
  `NULL`.

- `class_levels`: Character vector, the factor levels for
  classification, or `NULL`.

- `multistep_info`: For the single vector-valued output case only, a
  list with `steps` (integer vector) and `vars` (character vector)
  describing the structure of the outcome's columns. `NULL` otherwise.
