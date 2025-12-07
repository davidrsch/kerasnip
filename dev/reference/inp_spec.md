# Remap Layer Block Arguments for Model Specification

Creates a wrapper function around a Keras layer block to rename its
arguments. This is a powerful helper for defining the `layer_blocks` in
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/create_keras_functional_spec.md)
and
[`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/create_keras_sequential_spec.md),
allowing you to connect reusable blocks into a model graph without
writing verbose anonymous functions.

## Usage

``` r
inp_spec(block, input_map)
```

## Arguments

- block:

  A function that defines a Keras layer or a set of layers. The first
  arguments should be the input tensor(s).

- input_map:

  A single character string or a named character vector that specifies
  how to rename/remap the arguments of `block`.

## Value

A new function (a closure) that wraps the `block` function with renamed
arguments, ready to be used in a `layer_blocks` list.

## Details

`inp_spec()` makes your model definitions cleaner and more readable. It
handles the metaprogramming required to create a new function with the
correct argument names, while preserving the original block's
hyperparameters and their default values.

The function supports two modes of operation based on `input_map`:

1.  **Single Input Renaming**: If `input_map` is a single character
    string, the wrapper function renames the *first* argument of the
    `block` function to the provided string. This is the common case for
    blocks that take a single tensor input.

2.  **Multiple Input Mapping**: If `input_map` is a named character
    vector, the **names must match the argument names of `block`** and
    each value must be the name of an upstream layer block whose output
    should be fed into that argument. This orientation matches the
    syntax (e.g., `c(numeric = "processed_numerical")`). This is used
    for blocks with multiple inputs, like a concatenation layer.

*Note*: Prior releases accepted the opposite orientation
(`c(processed_numerical = "numeric")`). Existing code written in that
style must flip the names/values when upgrading to this version.

## Examples

``` r
# \donttest{
# --- Example Blocks ---
# A standard dense block with one input tensor and one hyperparameter.
dense_block <- function(tensor, units = 16) {
  tensor |> keras3::layer_dense(units = units, activation = "relu")
}

# A block that takes two tensors as input.
concat_block <- function(input_a, input_b) {
  keras3::layer_concatenate(list(input_a, input_b))
}

# An output block with one input.
output_block <- function(tensor) {
  tensor |> keras3::layer_dense(units = 1)
}

# --- Usage ---
layer_blocks <- list(
  main_input = keras3::layer_input,
  path_a = inp_spec(dense_block, "main_input"),
  path_b = inp_spec(dense_block, "main_input"),
  concatenated = inp_spec(
    concat_block,
    c(input_a = "path_a", input_b = "path_b")
  ),
  output = inp_spec(output_block, "concatenated")
)
# }
```
