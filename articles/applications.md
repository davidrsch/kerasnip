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
#>         0/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step     8192/170498071 ━━━━━━━━━━━━━━━━━━━━ 19:31 7us/step    32768/170498071 ━━━━━━━━━━━━━━━━━━━━ 9:51 3us/step     98304/170498071 ━━━━━━━━━━━━━━━━━━━━ 4:57 2us/step   212992/170498071 ━━━━━━━━━━━━━━━━━━━━ 3:03 1us/step   434176/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:52 1us/step   892928/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:05 0us/step  1794048/170498071 ━━━━━━━━━━━━━━━━━━━━ 37s 0us/step   3596288/170498071 ━━━━━━━━━━━━━━━━━━━━ 21s 0us/step  7184384/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step  9895936/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step  10551296/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 13885440/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 16703488/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 19652608/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 22814720/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 26083328/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 29261824/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 32169984/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 35340288/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 38371328/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 41435136/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 43343872/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 46235648/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 46817280/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 50372608/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 52314112/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 54272000/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 56229888/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 58171392/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 60137472/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 62119936/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 64110592/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 66052096/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 68001792/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 70000640/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 72024064/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 74047488/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 76054528/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 78020608/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 79233024/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 81043456/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 83099648/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 85155840/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 87203840/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 89251840/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 91283456/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 93011968/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 94732288/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 96755712/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 98459648/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step100278272/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step101990400/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step103866368/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step105979904/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step108077056/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step109821952/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step111681536/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step113287168/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step115343360/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step116940800/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step118915072/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step120848384/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step122839040/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step124690432/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step126500864/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step128335872/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step130138112/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step131948544/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step133963776/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step135929856/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step136871936/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step140353536/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step141713408/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step143040512/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step144580608/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step146112512/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step147693568/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step149258240/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step150831104/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step152395776/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step153968640/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step155574272/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step157171712/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step158777344/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step160382976/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step161980416/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step163577856/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step165216256/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step166854656/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step168435712/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step169664512/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step

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
#>        0/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step 2736128/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 9134080/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step15032320/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step21602304/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step27779072/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step34062336/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step40239104/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step45932544/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step52264960/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step58261504/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step64495616/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step70746112/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step76914688/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step83140608/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step89726976/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

# Evaluate on the test set
predictions <- predict(fit_functional, new_data = test_df_small)
#> 4/4 - 2s - 541ms/step
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
