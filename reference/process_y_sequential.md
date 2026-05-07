# Process Outcome Input for Keras

Preprocesses outcome data (`y`) into a format suitable for Keras models.
Handles both regression (numeric) and classification (factor) outcomes,
including one-hot encoding for classification.

## Usage

``` r
process_y_sequential(y, is_classification = NULL, class_levels = NULL)
```

## Arguments

- y:

  A vector of outcomes.

- is_classification:

  Logical, optional. If `TRUE`, treats `y` as classification. If
  `FALSE`, treats as regression. If `NULL` (default), it's determined
  from `is.factor(y)`.

- class_levels:

  Character vector, optional. The factor levels for classification
  outcomes. If `NULL` (default), determined from `levels(y)`.

## Value

A list containing:

- `y_proc`: The processed outcome data (matrix or one-hot encoded
  array).

- `is_classification`: Logical, indicating if `y` was treated as
  classification.

- `num_classes`: Integer, the number of classes for classification, or
  `NULL`.

- `class_levels`: Character vector, the factor levels for
  classification, or `NULL`.
