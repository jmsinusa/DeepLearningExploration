# https://www.tensorflow.org/tutorials/estimators/linear

# Overview: Using census data which contains data a person's age, education, marital status, and occupation
# (the features), we will try to predict whether or not the person earns more than 50,000 dollars a year
# (the target label). We will train a logistic regression model that, given an individual's information,
# outputs a number between 0 and 1—this can be interpreted as the probability that the individual has an annual
# income of over 50,000 dollars.

# ##############
# Setup
# ##############

import tensorflow as tf
import tensorflow.feature_column as fc

import os
import sys

import matplotlib.pyplot as plt

# Enable eager execution
tf.enable_eager_execution()

# ############
# Model
# ############
# https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

# Download from tensorflow's model repository
# https://github.com/tensorflow/models/


