# Transfer Learning with Keras Applications

## Introduction

Transfer learning is a powerful technique where a model developed for
one task is reused as the starting point for a model on a second task.
It is especially popular in computer vision, where pre-trained models
like `ResNet50`, which were trained on the massive ImageNet dataset, can
be used as powerful, ready-made feature extractors.

The `kerasnip` package makes it easy to incorporate these pre-trained
Keras Applications directly into a `tidymodels` workflow. This vignette
will demonstrate how to:

1.  Define a `kerasnip` model that uses a pre-trained `ResNet50` as a
    frozen base layer.
2.  Add a new, trainable classification “head” on top of the frozen
    base.
3.  Tune the hyperparameters of the new classification head using a
    standard `tidymodels` workflow.

## Setup

First, we load the necessary packages.

``` r
library(kerasnip)
library(tidymodels)
#> ── Attaching packages ────────────────────────────────────── tidymodels 1.4.1 ──
#> ✔ broom        1.0.11     ✔ recipes      1.3.1 
#> ✔ dials        1.4.2      ✔ rsample      1.3.1 
#> ✔ dplyr        1.1.4      ✔ tailor       0.1.0 
#> ✔ ggplot2      4.0.1      ✔ tidyr        1.3.1 
#> ✔ infer        1.0.9      ✔ tune         2.0.1 
#> ✔ modeldata    1.5.1      ✔ workflows    1.3.0 
#> ✔ parsnip      1.4.0      ✔ workflowsets 1.1.1 
#> ✔ purrr        1.2.0      ✔ yardstick    1.3.2
#> ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
#> ✖ purrr::discard() masks scales::discard()
#> ✖ dplyr::filter()  masks stats::filter()
#> ✖ dplyr::lag()     masks stats::lag()
#> ✖ recipes::step()  masks stats::step()
library(keras3)
#> 
#> Attaching package: 'keras3'
#> The following object is masked from 'package:yardstick':
#> 
#>     get_weights
```

## Data Preparation

We’ll use the CIFAR-10 dataset, which consists of 60,000 32x32 color
images in 10 classes. `keras3` provides a convenient function to
download it.

The `ResNet50` model was pre-trained on ImageNet, which has a different
set of classes. Our goal is to fine-tune it to classify the 10 classes
in CIFAR-10.

``` r
# Load CIFAR-10 dataset
cifar10 <- dataset_cifar10()
#> Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
#>         0/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step    98304/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:38 1us/step   860160/170498071 ━━━━━━━━━━━━━━━━━━━━ 22s 0us/step   4677632/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step  10289152/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 15949824/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 21536768/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 27082752/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 32792576/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 38420480/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 43950080/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 49561600/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 55263232/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 60858368/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 66412544/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 72032256/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 77651968/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step 83320832/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step 88743936/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step 94330880/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step 99991552/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step105611264/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step111132672/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step116736000/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step122347520/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step127983616/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step133586944/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step139272192/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step144875520/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step150528000/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step156147712/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step161775616/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step167452672/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step

# Separate training and test data
x_train <- cifar10$train$x
y_train <- cifar10$train$y
x_test <- cifar10$test$x
y_test <- cifar10$test$y

# Rescale pixel values from [0, 255] to [0, 1]
x_train <- x_train / 255
x_test <- x_test / 255

# Convert outcomes to factors for tidymodels
y_train_factor <- factor(y_train[, 1])
y_test_factor <- factor(y_test[, 1])

# For tidymodels, it's best to work with data frames.
# We'll use a list-column to hold the image arrays.
train_df <- tibble::tibble(
  x = lapply(seq_len(nrow(x_train)), function(i) x_train[i, , , , drop = TRUE]),
  y = y_train_factor
)

test_df <- tibble::tibble(
  x = lapply(seq_len(nrow(x_test)), function(i) x_test[i, , , , drop = TRUE]),
  y = y_test_factor
)

# Use a smaller subset for faster vignette execution
train_df_small <- train_df[1:500, ]
test_df_small <- test_df[1:100, ]
```

## Functional API with a Pre-trained Base

The standard approach for transfer learning is to use the Keras
Functional API. We will define a model where: 1. The base is a
pre-trained `ResNet50`, with its final classification layer removed
(`include_top = FALSE`). 2. The weights of the base are frozen
(`trainable = FALSE`) so that only our new layers are trained. 3. A new
classification “head” is added, consisting of a flatten layer and a
dense output layer.

### Define Layer Blocks

``` r
# Input block: shape is determined automatically from the data
input_block <- function(input_shape) {
  layer_input(shape = input_shape)
}

# ResNet50 base block
resnet_base_block <- function(tensor) {
  # The base model is not trainable; we use it for feature extraction.
  resnet_base <- application_resnet50(
    weights = "imagenet",
    include_top = FALSE
  )
  resnet_base$trainable <- FALSE
  resnet_base(tensor)
}

# New classification head
flatten_block <- function(tensor) {
  tensor |> layer_flatten()
}

output_block_functional <- function(tensor, num_classes) {
  tensor |> layer_dense(units = num_classes, activation = "softmax")
}
```

### Create the `kerasnip` Specification

We connect these blocks using
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/reference/create_keras_functional_spec.md).

``` r
create_keras_functional_spec(
  model_name = "resnet_transfer",
  layer_blocks = list(
    input = input_block,
    resnet_base = inp_spec(resnet_base_block, "input"),
    flatten = inp_spec(flatten_block, "resnet_base"),
    output = inp_spec(output_block_functional, "flatten")
  ),
  mode = "classification"
)
```

### Fit and Evaluate the Model

Now we can use our new `resnet_transfer()` specification within a
`tidymodels` workflow.

``` r
spec_functional <- resnet_transfer(
  fit_epochs = 5,
  fit_validation_split = 0.2
) |>
  set_engine("keras")

rec_functional <- recipe(y ~ x, data = train_df_small)

wf_functional <- workflow() |>
  add_recipe(rec_functional) |>
  add_model(spec_functional)

fit_functional <- fit(wf_functional, data = train_df_small)
#> Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
#>        0/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step 7553024/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step20365312/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step36511744/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step52371456/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step65585152/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step80445440/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step

# Evaluate on the test set
predictions <- predict(fit_functional, new_data = test_df_small)
#> 4/4 - 2s - 522ms/step
bind_cols(predictions, test_df_small) |>
  accuracy(truth = y, estimate = .pred_class)
#> # A tibble: 1 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy multiclass      0.13
```

Even with a small dataset and few epochs, the pre-trained features from
ResNet50 give us a reasonable starting point for accuracy.

## Conclusion

This vignette demonstrated how `kerasnip` bridges the world of
pre-trained Keras applications with the structured, reproducible
workflows of `tidymodels`.

The **Functional API** is the most direct way to perform transfer
learning by attaching a new head to a frozen base model.

This approach allows you to leverage the power of deep learning models
that have been trained on massive datasets, significantly boosting
performance on smaller, domain-specific tasks.
