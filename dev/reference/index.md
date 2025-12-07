# Package index

## Core Functions

Functions for generating and managing custom model specifications.

- [`create_keras_sequential_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/create_keras_sequential_spec.md)
  : Create a Custom Keras Sequential Model Specification for Tidymodels
- [`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/create_keras_functional_spec.md)
  : Create a Custom Keras Functional API Model Specification for
  Tidymodels
- [`inp_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/inp_spec.md)
  : Remap Layer Block Arguments for Model Specification
- [`remove_keras_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/remove_keras_spec.md)
  : Remove a Keras Model Specification and its Registrations

## Registration Helpers

Functions for registering custom Keras components like loss functions,
optimizers, and metrics.

- [`register_keras_loss()`](https://davidrsch.github.io/kerasnip/dev/reference/register_keras_loss.md)
  : Register a Custom Keras Loss
- [`register_keras_metric()`](https://davidrsch.github.io/kerasnip/dev/reference/register_keras_metric.md)
  : Register a Custom Keras Metric
- [`register_keras_optimizer()`](https://davidrsch.github.io/kerasnip/dev/reference/register_keras_optimizer.md)
  : Register a Custom Keras Optimizer
- [`keras_optimizers`](https://davidrsch.github.io/kerasnip/dev/reference/keras_objects.md)
  [`keras_losses`](https://davidrsch.github.io/kerasnip/dev/reference/keras_objects.md)
  [`keras_metrics`](https://davidrsch.github.io/kerasnip/dev/reference/keras_objects.md)
  : Dynamically Discovered Keras Objects

## Model Inspection and Evaluation

Functions for summarizing, evaluating, and extracting information from
trained Keras models.

- [`compile_keras_grid()`](https://davidrsch.github.io/kerasnip/dev/reference/compile_keras_grid.md)
  : Compile Keras Models Over a Grid of Hyperparameters
- [`extract_valid_grid()`](https://davidrsch.github.io/kerasnip/dev/reference/extract_valid_grid.md)
  : Extract Valid Grid from Compilation Results
- [`inform_errors()`](https://davidrsch.github.io/kerasnip/dev/reference/inform_errors.md)
  : Inform About Compilation Errors
- [`extract_keras_history()`](https://davidrsch.github.io/kerasnip/dev/reference/extract_keras_history.md)
  : Extract Keras Training History
- [`extract_keras_model()`](https://davidrsch.github.io/kerasnip/dev/reference/extract_keras_model.md)
  : Extract Keras Model from a Fitted Kerasnip Object
- [`keras_evaluate()`](https://davidrsch.github.io/kerasnip/dev/reference/keras_evaluate.md)
  : Evaluate a Kerasnip Model

## Custom recipe steps

Custom stpes for recipe which uses kerasnip models specifications

- [`step_collapse()`](https://davidrsch.github.io/kerasnip/dev/reference/step_collapse.md)
  : Collapse Predictors into a single list-column
