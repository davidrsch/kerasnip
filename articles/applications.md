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
#>         0/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step     8192/170498071 ━━━━━━━━━━━━━━━━━━━━ 19:51 7us/step    32768/170498071 ━━━━━━━━━━━━━━━━━━━━ 10:05 4us/step    98304/170498071 ━━━━━━━━━━━━━━━━━━━━ 5:06 2us/step    180224/170498071 ━━━━━━━━━━━━━━━━━━━━ 3:41 1us/step   303104/170498071 ━━━━━━━━━━━━━━━━━━━━ 2:45 1us/step   458752/170498071 ━━━━━━━━━━━━━━━━━━━━ 2:10 1us/step   696320/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:40 1us/step   999424/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:19 0us/step  1433600/170498071 ━━━━━━━━━━━━━━━━━━━━ 1:01 0us/step  1826816/170498071 ━━━━━━━━━━━━━━━━━━━━ 52s 0us/step   2260992/170498071 ━━━━━━━━━━━━━━━━━━━━ 46s 0us/step  2605056/170498071 ━━━━━━━━━━━━━━━━━━━━ 47s 0us/step  3612672/170498071 ━━━━━━━━━━━━━━━━━━━━ 36s 0us/step  4038656/170498071 ━━━━━━━━━━━━━━━━━━━━ 34s 0us/step  4464640/170498071 ━━━━━━━━━━━━━━━━━━━━ 32s 0us/step  4882432/170498071 ━━━━━━━━━━━━━━━━━━━━ 31s 0us/step  5300224/170498071 ━━━━━━━━━━━━━━━━━━━━ 30s 0us/step  5750784/170498071 ━━━━━━━━━━━━━━━━━━━━ 29s 0us/step  6193152/170498071 ━━━━━━━━━━━━━━━━━━━━ 28s 0us/step  6627328/170498071 ━━━━━━━━━━━━━━━━━━━━ 28s 0us/step  7077888/170498071 ━━━━━━━━━━━━━━━━━━━━ 27s 0us/step  7536640/170498071 ━━━━━━━━━━━━━━━━━━━━ 26s 0us/step  7979008/170498071 ━━━━━━━━━━━━━━━━━━━━ 26s 0us/step  8437760/170498071 ━━━━━━━━━━━━━━━━━━━━ 25s 0us/step  8904704/170498071 ━━━━━━━━━━━━━━━━━━━━ 25s 0us/step  9363456/170498071 ━━━━━━━━━━━━━━━━━━━━ 24s 0us/step  9830400/170498071 ━━━━━━━━━━━━━━━━━━━━ 24s 0us/step 10280960/170498071 ━━━━━━━━━━━━━━━━━━━━ 24s 0us/step 10747904/170498071 ━━━━━━━━━━━━━━━━━━━━ 23s 0us/step 11214848/170498071 ━━━━━━━━━━━━━━━━━━━━ 23s 0us/step 11657216/170498071 ━━━━━━━━━━━━━━━━━━━━ 23s 0us/step 12107776/170498071 ━━━━━━━━━━━━━━━━━━━━ 22s 0us/step 12558336/170498071 ━━━━━━━━━━━━━━━━━━━━ 22s 0us/step 13000704/170498071 ━━━━━━━━━━━━━━━━━━━━ 22s 0us/step 13434880/170498071 ━━━━━━━━━━━━━━━━━━━━ 22s 0us/step 13885440/170498071 ━━━━━━━━━━━━━━━━━━━━ 22s 0us/step 14344192/170498071 ━━━━━━━━━━━━━━━━━━━━ 21s 0us/step 14811136/170498071 ━━━━━━━━━━━━━━━━━━━━ 21s 0us/step 15278080/170498071 ━━━━━━━━━━━━━━━━━━━━ 21s 0us/step 15720448/170498071 ━━━━━━━━━━━━━━━━━━━━ 21s 0us/step 16187392/170498071 ━━━━━━━━━━━━━━━━━━━━ 21s 0us/step 16646144/170498071 ━━━━━━━━━━━━━━━━━━━━ 20s 0us/step 17104896/170498071 ━━━━━━━━━━━━━━━━━━━━ 20s 0us/step 17571840/170498071 ━━━━━━━━━━━━━━━━━━━━ 20s 0us/step 18038784/170498071 ━━━━━━━━━━━━━━━━━━━━ 20s 0us/step 18497536/170498071 ━━━━━━━━━━━━━━━━━━━━ 20s 0us/step 18964480/170498071 ━━━━━━━━━━━━━━━━━━━━ 20s 0us/step 19415040/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step 19857408/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step 20332544/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step 20791296/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step 21250048/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step 21708800/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step 22175744/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step 22618112/170498071 ━━━━━━━━━━━━━━━━━━━━ 19s 0us/step 23085056/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step 23552000/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step 23994368/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step 24494080/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step 24985600/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step 25477120/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step 25976832/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step 26525696/170498071 ━━━━━━━━━━━━━━━━━━━━ 18s 0us/step 27058176/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step 27598848/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step 28147712/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step 28688384/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step 29237248/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step 29794304/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step 30359552/170498071 ━━━━━━━━━━━━━━━━━━━━ 17s 0us/step 30867456/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 31408128/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 31965184/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 32481280/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 33030144/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 33513472/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 34037760/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 34537472/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 35053568/170498071 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 35561472/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 36044800/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 36487168/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 36921344/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 37388288/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 37855232/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 38330368/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 38805504/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 39256064/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 39747584/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 40206336/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 40689664/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 41205760/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step 41697280/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 42196992/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 42704896/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 43204608/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 43696128/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 44195840/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 44703744/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 45170688/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 45670400/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 46153728/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 46612480/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 47087616/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 47513600/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 47980544/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 48447488/170498071 ━━━━━━━━━━━━━━━━━━━━ 14s 0us/step 48889856/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 49356800/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 49807360/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 50249728/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 50716672/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 51175424/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 51658752/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 52224000/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 52756480/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 53280768/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 53829632/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 54353920/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 54870016/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 55377920/170498071 ━━━━━━━━━━━━━━━━━━━━ 13s 0us/step 55877632/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 56401920/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 56926208/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 57458688/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 58007552/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 58548224/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 59064320/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 59613184/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 60104704/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 60604416/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 61071360/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 61530112/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 61997056/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 62455808/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 62906368/170498071 ━━━━━━━━━━━━━━━━━━━━ 12s 0us/step 63373312/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 63832064/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 64282624/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 64749568/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 65200128/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 65667072/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 66134016/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 66592768/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 67059712/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 67510272/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 67952640/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 68419584/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 68886528/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 69345280/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 69804032/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 70254592/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 70696960/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 71147520/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 71598080/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 72040448/170498071 ━━━━━━━━━━━━━━━━━━━━ 11s 0us/step 72474624/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 72908800/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 73342976/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 73785344/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 74252288/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 74702848/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 75145216/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 75628544/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 76111872/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 76595200/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 77070336/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 77553664/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 78053376/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 78544896/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 79028224/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 79527936/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 80003072/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 80486400/170498071 ━━━━━━━━━━━━━━━━━━━━ 10s 0us/step 80977920/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step  81485824/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 81977344/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 82468864/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 82968576/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 83460096/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 83968000/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 84459520/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 84959232/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 85450752/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 85934080/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 86376448/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 86827008/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 87277568/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 87719936/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 88178688/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 88629248/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 89079808/170498071 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step 89554944/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 90087424/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 90644480/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 91185152/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 91734016/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 92274688/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 92839936/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 93405184/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 93986816/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 94625792/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 95256576/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 95911936/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 96542720/170498071 ━━━━━━━━━━━━━━━━━━━━ 8s 0us/step 97165312/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 97820672/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 98492416/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 99237888/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step 99975168/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step100737024/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step101466112/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step102170624/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step102916096/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step103645184/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step104349696/170498071 ━━━━━━━━━━━━━━━━━━━━ 7s 0us/step105103360/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step105824256/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step106545152/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step107298816/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step108019712/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step108732416/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step109436928/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step110166016/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step110870528/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step111575040/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step112427008/170498071 ━━━━━━━━━━━━━━━━━━━━ 6s 0us/step113319936/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step114180096/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step115073024/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step115933184/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step116760576/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step117637120/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step118489088/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step119324672/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step120176640/170498071 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step121077760/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step121929728/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step122806272/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step123682816/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step124534784/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step125345792/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step126238720/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step127098880/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step127959040/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step128786432/170498071 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step129646592/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step130473984/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step131522560/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step132636672/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step133554176/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step134684672/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step135815168/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step136863744/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step137936896/170498071 ━━━━━━━━━━━━━━━━━━━━ 3s 0us/step138936320/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step139902976/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step140886016/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step141860864/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step142934016/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step143843328/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step144728064/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step145817600/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step146841600/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step147841024/170498071 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step148676608/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step149471232/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step150413312/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step151461888/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step152207360/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step152993792/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step153763840/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step154599424/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step155402240/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step156409856/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step157310976/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step158146560/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step159080448/170498071 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step160063488/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step161136640/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step162226176/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step163160064/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step164167680/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step165150720/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step166264832/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step167239680/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step168132608/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step169132032/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170041344/170498071 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step170498071/170498071 ━━━━━━━━━━━━━━━━━━━━ 15s 0us/step

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
#>        0/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0s/step   81920/94765736 ━━━━━━━━━━━━━━━━━━━━ 59s 1us/step  319488/94765736 ━━━━━━━━━━━━━━━━━━━━ 30s 0us/step  884736/94765736 ━━━━━━━━━━━━━━━━━━━━ 16s 0us/step 2031616/94765736 ━━━━━━━━━━━━━━━━━━━━ 9s 0us/step  5242880/94765736 ━━━━━━━━━━━━━━━━━━━━ 4s 0us/step11894784/94765736 ━━━━━━━━━━━━━━━━━━━━ 2s 0us/step26279936/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step38797312/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step53968896/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step69926912/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step86491136/94765736 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 1s 0us/step

# Evaluate on the test set
predictions <- predict(fit_functional, new_data = test_df_small)
#> 4/4 - 2s - 609ms/step
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
