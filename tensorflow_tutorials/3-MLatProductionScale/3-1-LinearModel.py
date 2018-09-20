# https://www.tensorflow.org/tutorials/estimators/linear

# Overview: Using census data which contains data a person's age, education, marital status, and occupation
# (the features), we will try to predict whether or not the person earns more than 50,000 dollars a year
# (the target label). We will train a logistic regression model that, given an individual's information,
# outputs a number between 0 and 1—this can be interpreted as the probability that the individual has an annual
# income of over 50,000 dollars.

# ########################################
# Setup
# ########################################

import tensorflow as tf
import tensorflow.feature_column as fc

import os
import sys

import matplotlib.pyplot as plt

# Enable eager execution
tf.enable_eager_execution()

# ########################################
# Download model
# ########################################

# https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html

# Download from tensorflow's model repository
# https://github.com/tensorflow/models/

# git clone --depth 1 https://github.com/tensorflow/models

# Add models directory to path
models_path = os.path.join(os.getcwd(), 'models')
sys.path.append(models_path)

# ########################################
# Download dataset
# ########################################
from official.wide_deep import census_dataset
from official.wide_deep import census_main

census_dataset.download("/tmp/census_data/")
train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"

# ########################################
# Review dataset
# ########################################

import pandas

train_df = pandas.read_csv(train_file, header=None, names=census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header=None, names=census_dataset._CSV_COLUMNS)

train_df.head()


# ########################################
# Convert into tensors
# ########################################

# When building a tf.estimator model, the input data is specified by using an input function (or input_fn).
# This builder function returns a tf.data.Dataset of batches of (features-dict, label) pairs.
# It is not called until it is passed to tf.estimator.Estimator methods such as train and evaluate.

# The input builder function returns the following pair:
#
#     features: A dict from feature names to Tensors or SparseTensors containing batches of features.
#     labels: A Tensor containing batches of labels.
#
# The keys of the features are used to configure the model's input layer.

# Note: The input function is called while constructing the TensorFlow graph, not while running the graph.
# It is returning a representation of the input data as a sequence of TensorFlow graph operations.

# For small problems like this, it's easy to make a tf.data.Dataset by slicing the pandas.DataFrame:
def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
    label = df[label_key]
    ds = tf.data.Dataset.from_tensor_slices((dict(df), label))

    if shuffle:
        ds = ds.shuffle(10000)

    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds


ds = easy_input_function(train_df, label_key='income_bracket', num_epochs=5, shuffle=True, batch_size=10)

# Look at one  batch
for feature_batch, label_batch in ds.take(1):
    print('Some feature keys:', list(feature_batch.keys())[:5])
    print()
    print('A batch of Ages  :', feature_batch['age'])
    print()
    print('A batch of Labels:', label_batch)

# But this approach has severly-limited scalability. Larger datasets should be streamed from disk.
# The census_dataset.input_fn provides an example of how to do this using tf.decode_csv and tf.data.TextLineDataset:

ds = census_dataset.input_fn(train_file, num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
    print('Feature keys:', list(feature_batch.keys())[:5])
    print()
    print('Age batch   :', feature_batch['age'])
    print()
    print('Label batch :', label_batch)

# Because Estimators expect an input_fn that takes no arguments, we typically wrap configurable input function into an
# object with the expected signature. For this notebook configure the train_inpf to iterate over the data twice:

import functools

train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

# ########################################
# Selecting and engineering features
# ########################################

# Estimators use a system called feature columns to describe how the model should interpret each of the raw
# input features. An Estimator expects a vector of numeric inputs, and feature columns describe how the model
# should convert each feature.

# The simplest feature_column is numeric_column. This indicates that a feature is a numeric value that should be input
# to the model directly. For example:

age = fc.numeric_column('age')

# The model will use the feature_column definitions to build the model input.
# You can inspect the resulting output using the input_layer function:

fc.input_layer(feature_batch, [age]).numpy()

# Train a model just using age

classifier = tf.estimator.LinearClassifier(feature_columns=[age])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
print(result)
# 'auc_precision_recall': 0.3113661, 'precision': 0.21875, 'recall': 0.0018200728, 'accuracy': 0.76266813,

# Define a NumericColumn for each continuous feature column that we want to use in the model

education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

my_numeric_columns = [age, education_num, capital_gain, capital_loss, hours_per_week]

fc.input_layer(feature_batch, my_numeric_columns).numpy()

# Retrain the model
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))
# auc_precision_recall: 0.5738427; precision: 0.57384986; recall: 0.30811232; accuracy: 0.78250724

# Categorical columns
# If you know all categories, use categorical_column_with_vocabulary_list to one-hot encode each option
relationship = fc.categorical_column_with_vocabulary_list(
    'relationship',
    ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])
# Our network is expecting a dense encoding, so either use tf.feature_column.indicator_column
# or tf.feature_column.embedding_column.
fc.input_layer(feature_batch, [age, fc.indicator_column(relationship)])

# If you don't know all possible values in advance, use categorical_column_with_hash_bucket
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)
for item in feature_batch['occupation'].numpy():
    print(item.decode())
# View the layer we have crated. Note it is batch size, hash buck size, in this case (10, 1000)
occupation_result = fc.input_layer(feature_batch, [fc.indicator_column(occupation)])
occupation_result.numpy().shape


classifier = tf.estimator.LinearClassifier(feature_columns=[age, education_num, capital_gain, capital_loss,
                                                            hours_per_week, relationship, occupation])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))

education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])


my_categorical_columns = [relationship, occupation, education, marital_status, workclass]

# Configure a model using all of cols
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns+my_categorical_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

for key,value in sorted(result.items()):
    print('%s: %s' % (key, value))
    # accuracy: 0.82998586

# #####
# Derived feature columns

