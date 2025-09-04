#' Create a Custom Keras Sequential Model Specification for Tidymodels
#'
#' @description
#' This function acts as a factory to generate a new `parsnip` model
#' specification based on user-defined blocks of Keras layers using the
#' Sequential API. This is the ideal choice for creating models that are a
#' simple, linear stack of layers. For models with complex, non-linear
#' topologies, see [create_keras_functional_spec()].
#'
#' @param model_name A character string for the name of the new model
#'   specification function (e.g., "custom_cnn"). This should be a valid R
#'   function name.
#' @param layer_blocks A named, ordered list of functions. Each function defines
#'   a "block" of Keras layers. The function must take a Keras model object as
#'   its first argument and return the modified model. Other arguments to the
#'   function will become tunable parameters in the final model specification.
#' @param mode A character string, either "regression" or "classification".
#' @param ... Reserved for future use. Currently not used.
#' @param env The environment in which to create the new model specification
#'   function and its associated `update()` method. Defaults to the calling
#'   environment (`parent.frame()`).
#'
#' @details
#' This function generates all the boilerplate needed to create a custom,
#' tunable `parsnip` model specification that uses the Keras Sequential API.
#'
#' The function inspects the arguments of your `layer_blocks` functions
#' (ignoring special arguments like `input_shape` and `num_classes`)
#' and makes them available as arguments in the generated model specification,
#' prefixed with the block's name (e.g., `dense_units`).
#'
#' The new model specification function and its `update()` method are created in
#' the environment specified by the `env` argument.
#'
#' @section Model Architecture (Sequential API):
#' `kerasnip` builds the model by applying the functions in `layer_blocks` in
#' the order they are provided. Each function receives the Keras model built by
#' the previous function and returns a modified version.
#'
#' 1.  The **first block** must initialize the model (e.g., with
#'     `keras_model_sequential()`). It can accept an `input_shape` argument,
#'     which `kerasnip` will provide automatically during fitting.
#' 2.  **Subsequent blocks** add layers to the model.
#' 3.  The **final block** should add the output layer. For classification, it
#'     can accept a `num_classes` argument, which is provided automatically.
#'
#' A key feature of this function is the automatic creation of `num_{block_name}`
#' arguments (e.g., `num_hidden`). This allows you to control how many times
#' each block is repeated, making it easy to tune the depth of your network.
#'
#' @importFrom rlang enquos dots_list arg_match env_poke
#' @importFrom parsnip update_dot_check
#'
#' @return Invisibly returns `NULL`. Its primary side effect is to create a new
#'   model specification function (e.g., `my_mlp()`) in the specified
#'   environment and register the model with `parsnip` so it can be used within
#'   the `tidymodels` framework.
#'
#' @seealso [remove_keras_spec()], [parsnip::new_model_spec()],
#'   [create_keras_functional_spec()]
#'
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("keras3", quietly = TRUE)) {
#' library(keras3)
#' library(parsnip)
#' library(dials)
#'
#' # 1. Define layer blocks for a complete model.
#' # The first block must initialize the model. `input_shape` is passed automatically.
#' input_block <- function(model, input_shape) {
#'   keras_model_sequential(input_shape = input_shape)
#' }
#' # A block for hidden layers. `units` will become a tunable parameter.
#' hidden_block <- function(model, units = 32) {
#'   model |> layer_dense(units = units, activation = "relu")
#' }
#'
#' # The output block. `num_classes` is passed automatically for classification.
#' output_block <- function(model, num_classes) {
#'   model |> layer_dense(units = num_classes, activation = "softmax")
#' }
#'
#' # 2. Create the spec, providing blocks in the correct order.
#' create_keras_sequential_spec(
#' model_name = "my_mlp_seq_spec",
#'   layer_blocks = list(
#'     input = input_block,
#'     hidden = hidden_block,
#'     output = output_block
#'   ),
#'   mode = "classification"
#' )
#'
#' # 3. Use the newly created specification function!
#' # Note the new arguments `num_hidden` and `hidden_units`.
#' model_spec <- my_mlp_seq_spec(
#'   num_hidden = 2,
#'   hidden_units = 64,
#'   epochs = 10,
#'   learn_rate = 0.01
#' )
#'
#' print(model_spec)
#' remove_keras_spec("my_mlp_seq_spec")
#' }
#' }
create_keras_sequential_spec <- function(
  model_name,
  layer_blocks,
  mode = c("regression", "classification"),
  ...,
  env = parent.frame()
) {
  mode <- arg_match(mode)
  create_keras_spec_impl(
    model_name,
    layer_blocks,
    mode,
    functional = FALSE,
    env
  )
}
