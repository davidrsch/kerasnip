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
#>         0/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step     8192/170498071 ━━━━━━━━━━━━━━━━━━━━ 20:08 7us/step    32768/170498071 ━━━━━━━━━━━━━━━━━━━━ 10:07 4us/step    98304/170498071 ━━━━━━━━━━━━━━━━━━━━ 5:14 2us/step    204800/170498071 ━━━━━━━━━━━━━━━━━━━━ 3:18 1us/step   425984/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:58 1us/step   876544/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:09 0us/step  1769472/170498071 ━━━━━━━━━━━━━━━━━━━━ 39s 0us/step   3547136/170498071 ━━━━━━━━━━━━━━━━━━━━ 22s 0us/step  6873088/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step  9297920/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step  12271616/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 14368768/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 17293312/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 20480000/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 23584768/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 26755072/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 30015488/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 33292288/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 36470784/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 39510016/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 42696704/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 45793280/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 48857088/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 51986432/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 55181312/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 58318848/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 61513728/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 64749568/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 67911680/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 71032832/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 74153984/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 77381632/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 80388096/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 83542016/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 86458368/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 88760320/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 91611136/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 94855168/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 98099200/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step101302272/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step104472576/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step107716608/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step110804992/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step114016256/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step117219328/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step120635392/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step123797504/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step126935040/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step130195456/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step133455872/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step136560640/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step139763712/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step143015936/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step146186240/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step150069248/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step152829952/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step154951680/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step157491200/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step159539200/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step162570240/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step165683200/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step168861696/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step

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
[`create_keras_functional_spec()`](https://davidrsch.github.io/kerasnip/dev/reference/create_keras_functional_spec.md).

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
#>        0/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step  417792/94765736 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 2940928/94765736 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 13156352/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step22806528/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step32145408/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step41615360/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step49741824/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step59506688/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step69476352/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step78848000/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step88276992/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

# Evaluate on the test set
predictions <- predict(fit_functional, new_data = test_df_small)
#> 4/4 - 2s - 529ms/step
bind_cols(predictions, test_df_small) |>
  accuracy(truth = y, estimate = .pred_class)
#> # A tibble: 1 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy multiclass      0.25
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
