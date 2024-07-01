# Landmark Classification & Tagging for Social Media #
A landmark-classification model training comparison project for Udacity's AWS Machine Learning Fundamentals Nanodegree. 

## Table of Contents ##
- [Landmark Classification \& Tagging for Social Media](#landmark-classification--tagging-for-social-media)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [File Contents](#file-contents)
  - [Notebook Preview](#notebook-preview)
  - [Inferences](#inferences)


## Overview<a name="overview"></a> ##
This is a project in Udacity’s AWS Machine Learning Fundamentals Nanodegree comparing the training of data between a simpler CNN model from scratch and a pre-trained deep neural network model.

Two machine learning models: a *CNN with no pre-trained data*, and a *deep neural network with pre-trained data*, are put to the test to identify and classify landmarks based off of pictures.

The dataset consists of pictures from 50 labelled landmarks and separated into `test` and `train` for training validation.

The goal of this project is to compare the *performance* of the two models and the *convenience* between using these models.

## File Contents<a name="file_contents"></a> ##
* `src\` : Directory for the APIs for the learning models

* `static_images\` : Image directory

* `app_files` : Static image files for `app.html` snapshot

* `app.html` `app.ipynb` : Jupyter notebook (and HTML) for web app frontend

* `cnn_from_scratch.*` : Jupyter notebook (and HTML) for demonstration of training on *CNN from scratch*

* `transfer_learning.*` : Jupyter notebook (and HTML) for demonstration of training on *transfer learning*

## Notebook Preview<a name="notebook_preview"></a> ##
Snapshots of the Jupyter notebook is saved as HTML for ease of viewing.  
1. [App Frontend (Frozen)](https://ishanmitra.github.io/landmark-classification-cnn/app.html)
2. [CNN from scratch](https://ishanmitra.github.io/landmark-classification-cnn/cnn_from_scratch.html)
3. [Transfer Learning](https://ishanmitra.github.io/landmark-classification-cnn/transfer_learning.html)


## Inferences<a name="inferences"></a> ##
The CNN model test accuracy is **56%** (697/1250) whereas the test accuracy on a pre-trained ResNet18 model is **74%** (909/1250).

The advantage of pre-trained weights for deep neural networks is that the model is quickly able to identify features vital to identify landmarks accurately and predict the location.

The CNN model is a simpler network that was able to classify more than half of the images correctly without any pre-trained weights. The model has a drawback of not being able to converge any further.

The ResNet would require a lot of training time and data to generate the weights to achieve a similar accuracy. However, transfer learning removes that inconvenience allowing the model to train on a custom dataset.