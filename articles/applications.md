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
#> ✔ broom        1.0.12     ✔ recipes      1.3.2 
#> ✔ dials        1.4.3      ✔ rsample      1.3.2 
#> ✔ dplyr        1.2.1      ✔ tailor       0.1.0 
#> ✔ ggplot2      4.0.2      ✔ tidyr        1.3.2 
#> ✔ infer        1.1.0      ✔ tune         2.0.1 
#> ✔ modeldata    1.5.1      ✔ workflows    1.3.0 
#> ✔ parsnip      1.5.0      ✔ workflowsets 1.1.1 
#> ✔ purrr        1.2.2      ✔ yardstick    1.4.0
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
#> The following object is masked from 'package:infer':
#> 
#>     generate
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
#>         0/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step    98304/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:35 1us/step   442368/170498071 ━━━━━━━━━━━━━━━━━━━━ 43s 0us/step   1155072/170498071 ━━━━━━━━━━━━━━━━━━━━ 23s 0us/step  2015232/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step  2818048/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step  3710976/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step  4538368/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step  5439488/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step  6275072/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step  7151616/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step  7938048/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step  8773632/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step  9601024/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 10436608/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 11329536/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 12115968/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 13000704/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 13836288/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 14737408/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 15564800/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 16392192/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 17244160/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 18071552/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step  18964480/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 19791872/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 20692992/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 21520384/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 22421504/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 23248896/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 24158208/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 24985600/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 25886720/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 26714112/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 27549696/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 28442624/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 29278208/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 30179328/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 31006720/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 31907840/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 32718848/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 33570816/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 34406400/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 35299328/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 36126720/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 36855808/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 37675008/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 38510592/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 39403520/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 40239104/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 41066496/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 41836544/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 42606592/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 43442176/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 44285952/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 45105152/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 45842432/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 46579712/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 47292416/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 48054272/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 48898048/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 49725440/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 50487296/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 51257344/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 52027392/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 52731904/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 53501952/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 54272000/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 55164928/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 55943168/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 56705024/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 57417728/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 58179584/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 58949632/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 59777024/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 60612608/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 61382656/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 62152704/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 62857216/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 63692800/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 64651264/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 65609728/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 66461696/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 67272704/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 68124672/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 68943872/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 69779456/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 70737920/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 71639040/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 72466432/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 73318400/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 74170368/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 74964992/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 75800576/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 76767232/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 77668352/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 78495744/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 79388672/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 80297984/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 81313792/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 82919424/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 85098496/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 86319104/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 87597056/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 88940544/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 90161152/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 91701248/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 93552640/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 95232000/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 98131968/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step101089280/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step103243776/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step105545728/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step108503040/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step112058368/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step114458624/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step117276672/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step119775232/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step122855424/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step124960768/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step127655936/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step130670592/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step133234688/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step135282688/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step138289152/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step140763136/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step143654912/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step147521536/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step151470080/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step154574848/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step157917184/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step162111488/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step165470208/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step168763392/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step

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
#>        0/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step  786432/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 1540096/94765736 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 2359296/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 3080192/94765736 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 3899392/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 4423680/94765736 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 5079040/94765736 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 5832704/94765736 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 6488064/94765736 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 7667712/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 8617984/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 9469952/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step10682368/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step11796480/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step12746752/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step14057472/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step15073280/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step15761408/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step16941056/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step17629184/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step18382848/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step19070976/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step19873792/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step21528576/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step22282240/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step23363584/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step28311552/94765736 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step36888576/94765736 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step39567360/94765736 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step44138496/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step46661632/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step47824896/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step50053120/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step51101696/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step52117504/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step52936704/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step54214656/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step55328768/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step56475648/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step58310656/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step60735488/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step61849600/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step62963712/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step63586304/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step64135168/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step65126400/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step67354624/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step68239360/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step70795264/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step72597504/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step74072064/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step75218944/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step76136448/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step81248256/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step82558976/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step83443712/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step84885504/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step89505792/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step90718208/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step92028928/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step92979200/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step93732864/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step

# Evaluate on the test set
predictions <- predict(fit_functional, new_data = test_df_small)
#> 4/4 - 2s - 594ms/step
bind_cols(predictions, test_df_small) |>
  accuracy(truth = y, estimate = .pred_class)
#> # A tibble: 1 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy multiclass      0.16
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
