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
#>         0/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step    98304/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:31 1us/step   450560/170498071 ━━━━━━━━━━━━━━━━━━━━ 39s 0us/step    999424/170498071 ━━━━━━━━━━━━━━━━━━━━ 26s 0us/step  1548288/170498071 ━━━━━━━━━━━━━━━━━━━━ 22s 0us/step  2121728/170498071 ━━━━━━━━━━━━━━━━━━━━ 20s 0us/step  2719744/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step  3325952/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step  3932160/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step  4538368/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step  5144576/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step  5750784/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step  6348800/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step  6946816/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step  7544832/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step  8151040/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step  8740864/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step  9338880/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step  9936896/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 10551296/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 11141120/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 11747328/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 12353536/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 12951552/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 13557760/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 14172160/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 14729216/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 15286272/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 15826944/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 16359424/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 16867328/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 17391616/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 17932288/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 18440192/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 18972672/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 19529728/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 20078592/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 20627456/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 21168128/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 21725184/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 22274048/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 22806528/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 23347200/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 23896064/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 24453120/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 24993792/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 25542656/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 26083328/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 26624000/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 27181056/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 27738112/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 28278784/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 28811264/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 29351936/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 29917184/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 30457856/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 31006720/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 31555584/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 32112640/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 32710656/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 33316864/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 33923072/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 34529280/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 35143680/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 35749888/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 36364288/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 36978688/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 37584896/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 38207488/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 38805504/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 39419904/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 40042496/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 40648704/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 41230336/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 41836544/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 42434560/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 43032576/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 43671552/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 44269568/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 44867584/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 45481984/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 46096384/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 46710784/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 47300608/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 47890432/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 48562176/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 49242112/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 49913856/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 50552832/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 51200000/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 51838976/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 52477952/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 53116928/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 53805056/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 54484992/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 55156736/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 55828480/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step  56516608/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 57196544/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 57909248/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 58695680/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 59482112/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 60276736/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 61071360/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 61874176/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 62668800/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 63430656/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 64200704/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 65019904/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 65896448/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 66682880/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 67477504/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 68263936/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 69074944/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 69869568/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 70639616/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 71450624/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 72245248/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 73039872/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 73777152/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 74588160/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 75382784/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 76169216/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 76963840/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 77733888/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 78503936/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 79290368/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 80101376/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 80846848/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 81575936/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 82305024/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 83116032/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 83886080/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 84590592/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 85254144/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 86016000/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 86827008/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 87556096/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 88236032/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 88924160/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 89718784/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 90480640/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 91136000/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 91824128/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 92594176/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 93339648/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 94027776/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step 94756864/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 95526912/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 96313344/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 97009664/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 97681408/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 98426880/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 99155968/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step 99794944/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step100433920/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step101097472/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step101777408/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step102416384/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step103063552/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step103751680/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step104554496/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step105324544/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step106078208/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step106889216/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step107659264/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step108494848/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step109199360/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step109985792/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step110731264/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step111566848/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step112304128/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step113106944/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step113819648/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step114581504/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step115408896/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step116137984/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step116883456/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step117612544/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step118439936/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step119177216/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step119939072/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step120668160/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step121479168/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step122306560/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step123248640/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step124198912/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step125198336/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step126410752/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step127631360/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step129015808/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step130228224/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step131629056/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step132792320/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step134144000/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step135249920/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step136790016/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step137854976/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step139075584/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step140197888/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step141623296/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step142860288/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step144138240/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step145227776/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step146505728/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step147857408/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step148996096/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step150183936/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step151445504/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step153051136/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step154648576/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step156655616/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step158523392/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step160874496/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step162553856/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step164610048/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step166068224/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step168452096/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step169795584/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step

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
#>        0/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step 4358144/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step 9592832/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step14934016/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step21725184/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step28254208/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step32456704/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step38518784/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step43950080/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step50126848/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step56123392/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step60702720/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step65961984/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step72417280/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step77004800/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step81436672/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step86450176/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step90357760/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

# Evaluate on the test set
predictions <- predict(fit_functional, new_data = test_df_small)
#> 4/4 - 2s - 555ms/step
bind_cols(predictions, test_df_small) |>
  accuracy(truth = y, estimate = .pred_class)
#> # A tibble: 1 × 3
#>   .metric  .estimator .estimate
#>   <chr>    <chr>          <dbl>
#> 1 accuracy multiclass      0.18
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
