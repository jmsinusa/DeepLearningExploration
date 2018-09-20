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
classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns + my_categorical_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))
    # prec: 0.64828515; recall: 0.50130004; auc_precision_recall: 0.6048239

# #####
# Derived feature columns

# ## Bucketisation
# Bucketisation: Split a continuous feature into buckets, eg age:
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
# See the one hot encoding:
fc.input_layer(feature_batch, [age, age_buckets]).numpy()
# prec: 0.61956; recall: 0.55850; auc_precision_recall: 0.60481083 ## Seems to have gotten worse.

# ## Crossed feature columns
education_x_occupation = tf.feature_column.crossed_column(
    ['education', 'occupation'], hash_bucket_size=1000)
age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)

classifier = tf.estimator.LinearClassifier(
    feature_columns=my_numeric_columns + my_categorical_columns + [education_x_occupation,
                                                                   age_buckets_x_education_x_occupation])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))
    # prec: 0.6096511; recall: 0.69968796; auc_precision_recall: 0.67998374; accuracy: 0.82322955

# ################################
# Define and train a model
# ################################

# # Define model

import tempfile

base_columns = [
    education, marital_status, relationship, workclass, occupation,
    age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]

model = tf.estimator.LinearClassifier(
    model_dir=tempfile.mkdtemp(),
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(learning_rate=0.1))

# # Train
train_inpf = functools.partial(census_dataset.input_fn, train_file,
                               num_epochs=40, shuffle=True, batch_size=64)

model.train(train_inpf)

# # Score
results = model.evaluate(test_inpf)
for key, value in sorted(result.items()):
    print('%s: %0.4f' % (key, value))
    # precision: 0.6097; recall: 0.6997; auc_precision_recall: 0.6800; accuracy: 0.8232

# ###############################
# Predict some values
# ###############################

import numpy as np

predict_df = test_df[:20].copy()

pred_iter = model.predict(
    lambda: easy_input_function(predict_df, label_key='income_bracket',
                                num_epochs=1, shuffle=False, batch_size=10))
classes = np.array(['<=50K', '>50K'])
pred_class_id = []
for pred_dict in pred_iter:
    pred_class_id.append(pred_dict['class_ids'])
predict_df['predicted_class'] = classes[np.array(pred_class_id)]
predict_df['correct'] = predict_df['predicted_class'] == predict_df['income_bracket']
predict_df[['income_bracket', 'predicted_class', 'correct']]

# #############################
# Add regularisation to prevent overfitting
# #############################

# L1 regularisation

model_l1 = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=10.0,
        l2_regularization_strength=0.0))

model_l1.train(train_inpf)

results = model_l1.evaluate(test_inpf)
for key in sorted(results):
    print('%s: %0.4f' % (key, results[key]))
    # precision: 0.6833; recall: 0.5671; auc_precision_recall: 0.6940; accuracy: 0.8356

# L2 regularisation

model_l2 = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=10.0))

model_l2.train(train_inpf)

results = model_l2.evaluate(test_inpf)
for key in sorted(results):
    print('%s: %0.4f' % (key, results[key]))
    # precision: 0.6921; recall: 0.5528; auc_precision_recall: 0.6943; accuracy: 0.8363


# Look at the weights
def get_flat_weights(model):
    weight_names = [
        name for name in model.get_variable_names()
        if "linear_model" in name and "Ftrl" not in name]

    weight_values = [model.get_variable_value(name) for name in weight_names]

    weights_flat = np.concatenate([item.flatten() for item in weight_values], axis=0)

    return weights_flat

weights_flat = get_flat_weights(model)
weights_flat_l1 = get_flat_weights(model_l1)
weights_flat_l2 = get_flat_weights(model_l2)

# Lots of zeros caused by unused hashes. Mask these out
weight_mask = weights_flat != 0

weights_base = weights_flat[weight_mask]
weights_l1 = weights_flat_l1[weight_mask]
weights_l2 = weights_flat_l2[weight_mask]

# Plot distn of weights
plt.figure()
_ = plt.hist(weights_base, bins=np.linspace(-3,3,30))
plt.title('Base Model')
plt.ylim([0,500])

plt.figure()
_ = plt.hist(weights_l1, bins=np.linspace(-3,3,30))
plt.title('L1 - Regularization')
plt.ylim([0,500])

plt.figure()
_ = plt.hist(weights_l2, bins=np.linspace(-3,3,30))
plt.title('L2 - Regularization')
_=plt.ylim([0,500])

