# CIFAR-10 Image Classification with a Custom CNN

This project implements a Convolutional Neural Network (CNN) from scratch using PyTorch to classify images from the CIFAR-10 dataset. The network is trained to classify images into one of 10 classes, including airplanes, cars, birds, cats, and more. Additionally, the project demonstrates how to preprocess and classify custom images using the trained network.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Custom Image Inference](#custom-image-inference)

## Overview

The goal of this project is to train a CNN for image classification on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes. The project includes training, evaluation, and inference scripts, allowing you to train the model on the CIFAR-10 dataset and predict classes for custom images.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.x
- PyTorch
- Torchvision
- PIL (Pillow)
- NumPy

You can install the required libraries using pip:


-pip install torch torchvision pillow numpy

## Dataset
The CIFAR-10 dataset is automatically downloaded using the torchvision.datasets.CIFAR10 class. The dataset includes 50,000 training images and 10,000 test images across 10 classes:

-Airplane

-Car

-Bird

-Cat

-Deer

-Dog

-Frog

-Horse

-Ship

-Truck

## Model Architecture
The model is a simple Convolutional Neural Network (CNN) built with the following architecture:

-Conv2D: 3 input channels (RGB), 12 output channels, kernel size 5

-MaxPool2D: Kernel size 2x2

-Conv2D: 12 input channels, 24 output channels, kernel size 5

-MaxPool2D: Kernel size 2x2

-Fully Connected Layer: 2455 inputs, 120 outputs

-Fully Connected Layer: 120 inputs, 84 outputs

-Fully Connected Layer: 84 inputs, 10 outputs (for 10 CIFAR-10 classes)

The network uses ReLU activations, and CrossEntropyLoss is used as the loss function, with Stochastic Gradient Descent (SGD) as the optimizer.

## Training the Model
To train the model on CIFAR-10, the train.py script follows these steps:

-Data Loading: The training and testing data are loaded using torchvision.datasets.CIFAR10.

-Model Definition: The CNN model is defined using the architecture described above.

-Training: The model is trained over 30 epochs using the SGD optimizer with a learning rate of 0.001 and momentum of 0.9.

-Loss Calculation: The loss is calculated using nn.CrossEntropyLoss.

-Saving the Model: After training, the model can be saved for later inference.

## Training process:


After training, the model is evaluated on the test dataset. The accuracy of the model is printed after evaluation.

Example accuracy output:

Accuracy: 68.75%

-The test data is used to calculate the model's performance using the following metrics:

-Correct Predictions: Number of correct classifications.

-Accuracy: Total percentage of correct classifications.

## Custom Image Inference
To classify custom images using the trained model, use the classify.py script. Custom images are preprocessed with the same transformations used during training (resizing, normalization), and the model predicts their class.

Example image inference:


You can test with the following images:


['download.jpeg', 'download.webp', 'aeroplane.jpeg', 'car.jpeg', 'horse.jpeg']

Sample output:


Prediction: airplane

Prediction: car

Prediction: horse
