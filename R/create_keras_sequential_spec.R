#' Create a Custom Keras Model Specification for Tidymodels
#'
#' This function acts as a factory to generate a new `parsnip` model
#' specification based on user-defined blocks of Keras layers. This allows for
#' creating complex, tunable architectures that integrate seamlessly with the
#' `tidymodels` ecosystem.
#'
#' @param model_name A character string for the name of the new model
#'   specification function (e.g., "custom_cnn"). This should be a valid R
#'   function name.
#' @param layer_blocks A named list of functions. Each function defines a "block"
#'   of Keras layers. The function must take a Keras model object as its first
#'   argument and return the modified model. Other arguments to the function
#'   will become tunable parameters in the final model specification.
#' @param mode A character string, either "regression" or "classification".
#' @param ... Reserved for future use. Currently not used.
#' @param env The environment in which to create the new model specification
#'   function and its associated `update()` method. Defaults to the calling
#'   environment (`parent.frame()`).
#' @importFrom rlang enquos dots_list arg_match env_poke
#' @importFrom parsnip update_dot_check
#'
#' @details
#' The user is responsible for defining the entire model architecture by providing
#' an ordered list of layer block functions.
#' 1. The first block function must initialize the model (e.g., with
#' \code{keras_model_sequential()}). It can accept an \code{input_shape} argument,
#'  which will be provided automatically by the fitting engine.
#' 2. Subsequent blocks add hidden layers.
#' 3. The final block should add the output layer. For classification, it can
#' accept a \code{num_classes} argument, which is provided automatically.
#'
#' The \code{create_keras_sequential_spec()} function will inspect the arguments of your
#' \code{layer_blocks} functions (ignoring \code{input_shape} and \code{num_classes})
#' and make them available as arguments in the generated model specification,
#' prefixed with the block's name (e.g.,
#' `dense_units`).
#'
#' It also automatically creates arguments like `num_dense` to control how many
#' times each block is repeated. In addition, common training parameters such as
#' `epochs`, `learn_rate`, `validation_split`, and `verbose` are added to the
#' specification.
#'
#' The new model specification function and its `update()` method are created in
#' the environment specified by the `env` argument.
#'
#' @return Invisibly returns `NULL`. Its primary side effect is to create a new
#'   model specification function (e.g., `dynamic_mlp()`) in the specified
#'   environment and register the model with `parsnip` so it can be used within
#'   the `tidymodels` framework.
#'
#' @seealso [remove_keras_spec()], [parsnip::new_model_spec()]
#'
#' @export
#' @examples
#' \dontrun{
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
#' model_name = "my_mlp",
#'   layer_blocks = list(
#'     input = input_block,
#'     hidden = hidden_block,
#'     output = output_block
#'   ),
#'   mode = "classification"
#' )
#'
#' # 3. Use the newly created specification function!
# Note the new arguments `num_hidden` and `hidden_units`.
#' model_spec <- my_mlp(
#'   num_hidden = 2,
#'   hidden_units = 64,
#'   epochs = 10,
#'   learn_rate = 0.01
#' )
#'
#' print(model_spec)
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
