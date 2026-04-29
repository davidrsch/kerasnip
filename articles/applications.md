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
#> ── Attaching packages ────────────────────────────────────── tidymodels 1.5.0 ──
#> ✔ broom        1.0.12     ✔ recipes      1.3.2 
#> ✔ dials        1.4.3      ✔ rsample      1.3.2 
#> ✔ dplyr        1.2.1      ✔ tailor       0.1.0 
#> ✔ ggplot2      4.0.3      ✔ tidyr        1.3.2 
#> ✔ infer        1.1.0      ✔ tune         2.1.0 
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
#>         0/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step    65536/170498071 ━━━━━━━━━━━━━━━━━━━━ 2:11 1us/step   286720/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:06 0us/step   581632/170498071 ━━━━━━━━━━━━━━━━━━━━ 47s 0us/step    868352/170498071 ━━━━━━━━━━━━━━━━━━━━ 41s 0us/step  1196032/170498071 ━━━━━━━━━━━━━━━━━━━━ 37s 0us/step  1564672/170498071 ━━━━━━━━━━━━━━━━━━━━ 33s 0us/step  1974272/170498071 ━━━━━━━━━━━━━━━━━━━━ 31s 0us/step  2457600/170498071 ━━━━━━━━━━━━━━━━━━━━ 28s 0us/step  3006464/170498071 ━━━━━━━━━━━━━━━━━━━━ 25s 0us/step  3604480/170498071 ━━━━━━━━━━━━━━━━━━━━ 23s 0us/step  4325376/170498071 ━━━━━━━━━━━━━━━━━━━━ 21s 0us/step  5013504/170498071 ━━━━━━━━━━━━━━━━━━━━ 20s 0us/step  5890048/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step  6873088/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step  7897088/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step  9142272/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 10444800/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 11862016/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 13492224/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 15286272/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 17301504/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step  19341312/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 21585920/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 23511040/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 25927680/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 28966912/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 31268864/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 33964032/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 36143104/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 38903808/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 41590784/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 43704320/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 45817856/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 46137344/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 48078848/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 50020352/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 51773440/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 53288960/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 55140352/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 56885248/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 58720256/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 60588032/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 62193664/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 64020480/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 65363968/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 67067904/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 68935680/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 70656000/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 72474624/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 74317824/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 76169216/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 77914112/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 79626240/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 81379328/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 83156992/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 84819968/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 86392832/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 88031232/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 89694208/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 91234304/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 92643328/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 94183424/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 95780864/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 97148928/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 98467840/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 99753984/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step101097472/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step102506496/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step103661568/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step104816640/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step106102784/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step107446272/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step108658688/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step109789184/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step110952448/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step112254976/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step113532928/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step114622464/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step115646464/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step116932608/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step118169600/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step119365632/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step120324096/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step121348096/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step122650624/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step123781120/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step124878848/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step125845504/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step126820352/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step127909888/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step129122304/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step130187264/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step131178496/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step132136960/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step133332992/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step134569984/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step135593984/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step136593408/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step137625600/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step138878976/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step140083200/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step141082624/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step142065664/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step143097856/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step144384000/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step145547264/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step146489344/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step147488768/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step148553728/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step149864448/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step151109632/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step152231936/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step153354240/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step154673152/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step155893760/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step156876800/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step157941760/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step159055872/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step160309248/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step161529856/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step162480128/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step163643392/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step164732928/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step165969920/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step167059456/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step168198144/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step169181184/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170311680/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step

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
#>        0/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step  933888/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 9199616/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step22716416/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step31678464/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step47439872/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step53968896/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step65134592/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step77078528/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step91430912/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

# Evaluate on the test set
predictions <- predict(fit_functional, new_data = test_df_small)
#> 4/4 - 2s - 622ms/step
bind_cols(predictions, test_df_small) |>
  accuracy(truth = y, estimate = .pred_class)
#> # A tibble: 1 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy multiclass      0.19
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
