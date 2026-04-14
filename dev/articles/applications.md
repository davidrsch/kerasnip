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
#>         0/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step     8192/170498071 ━━━━━━━━━━━━━━━━━━━━ 19:38 7us/step    32768/170498071 ━━━━━━━━━━━━━━━━━━━━ 9:53 3us/step     98304/170498071 ━━━━━━━━━━━━━━━━━━━━ 4:57 2us/step   180224/170498071 ━━━━━━━━━━━━━━━━━━━━ 3:40 1us/step   294912/170498071 ━━━━━━━━━━━━━━━━━━━━ 2:47 1us/step   458752/170498071 ━━━━━━━━━━━━━━━━━━━━ 2:08 1us/step   679936/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:41 1us/step  1007616/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:17 0us/step  1433600/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:01 0us/step  2064384/170498071 ━━━━━━━━━━━━━━━━━━━━ 47s 0us/step   2899968/170498071 ━━━━━━━━━━━━━━━━━━━━ 36s 0us/step  4128768/170498071 ━━━━━━━━━━━━━━━━━━━━ 28s 0us/step  5742592/170498071 ━━━━━━━━━━━━━━━━━━━━ 21s 0us/step  8175616/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 11124736/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 14180352/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 15908864/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step  18407424/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 20824064/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 22921216/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 25477120/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 27811840/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 30261248/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 31891456/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 33914880/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 36405248/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 38526976/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 40861696/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 43261952/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 45391872/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 47013888/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 49233920/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 51617792/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 53600256/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step 56033280/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 58474496/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 61079552/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 62709760/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 64389120/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 66904064/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 69222400/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 71155712/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 73588736/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step 75882496/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 77537280/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 79716352/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 82206720/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 84434944/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 86736896/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 89112576/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 91529216/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 93577216/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 95182848/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 97427456/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step 99794944/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step101703680/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step103809024/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step106192896/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step108249088/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step109846528/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step111443968/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step113631232/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step115867648/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step118169600/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step120602624/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step122830848/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step124452864/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step126025728/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step128344064/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step130760704/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step133054464/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step135462912/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step137781248/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step139378688/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step140976128/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step143261696/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step145588224/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step147898368/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step150306816/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step152100864/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step154238976/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step155901952/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step158081024/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step160448512/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step163143680/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step166092800/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step168763392/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step

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
#>        0/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step 6037504/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step15417344/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step24788992/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step33906688/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step43663360/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step53264384/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step62324736/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step71458816/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step79642624/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step89227264/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

# Evaluate on the test set
predictions <- predict(fit_functional, new_data = test_df_small)
#> 4/4 - 2s - 531ms/step
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
