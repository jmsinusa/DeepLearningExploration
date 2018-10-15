# https://www.tensorflow.org/hub/tutorials/text_classification_with_tf_hub

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


# ###############################
# Data
# ###############################

# We will try to solve the Large Movie Review Dataset v1.0 task from Mass et al.
# The dataset consists of IMDB movie reviews labeled by positivity from 1 to 10.
# The task is to label the reviews as negative or positive.

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    data["sentiment"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            data["sentiment"].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(os.path.join(directory, "pos"))
    neg_df = load_directory_data(os.path.join(directory, "neg"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname="aclImdb.tar.gz",
        origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        extract=True)

    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                         "aclImdb", "train"))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        "aclImdb", "test"))

    return train_df, test_df


# Reduce logging output.
tf.logging.set_verbosity(tf.logging.ERROR)

train_df, test_df = download_and_load_datasets()
train_df.head()

# Inport into tensorflow

# Training input on the whole training set with no limit on training epochs.
train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], num_epochs=None, shuffle=True)

# Prediction on the whole training set.
predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
    train_df, train_df["polarity"], shuffle=False)

# Prediction on the test set.
predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
    test_df, test_df["polarity"], shuffle=False)

# #######################
# Feature columns
# #######################

# TF-Hub provides a feature column that applies a module on the given text feature and passes further the outputs
# of the module. In this tutorial we will be using the nnlm-en-dim128 module. For the purpose of this tutorial,
# the most important facts are:
#
#     The module takes a batch of sentences in a 1-D tensor of strings as input.
#     The module is responsible for preprocessing of sentences (e.g. removal of punctuation and splitting on spaces).
#     The module works with any input (e.g. nnlm-en-dim128 hashes words not present in vocabulary into ~20.000 buckets).

embedded_text_feature_column = hub.text_embedding_column(
    key="sentence",
    module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

# #######################
# Estimator
# #######################

# Using a DNN classifier

estimator = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    feature_columns=[embedded_text_feature_column],
    n_classes=2,
    optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

# Training for 1,000 steps means 128,000 training examples with the default
# batch size. This is roughly equivalent to 5 epochs since the training dataset contains 25,000 examples.
estimator.train(input_fn=train_input_fn, steps=1000)

train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

print("Training set accuracy: {accuracy}".format(**train_eval_result))
print("Test set accuracy: {accuracy}".format(**test_eval_result))


# #######################
# Confusion matrix
# #######################

def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]


LABELS = ["negative", "positive"]

# Create a confusion matrix on training data.
with tf.Graph().as_default():
    cm = tf.confusion_matrix(train_df["polarity"],
                             get_predictions(estimator, predict_train_input_fn))
    with tf.Session() as session:
        cm_out = session.run(cm)

# Normalize the confusion matrix so that each row sums to 1.
cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS);
plt.xlabel("Predicted")
plt.ylabel("True")


# #######################
# Transfer learning
# #######################

# In this part, we will demonstrate this by training with two different TF-Hub modules:
#
#     nnlm-en-dim128 - pretrained text embedding module,
#     random-nnlm-en-dim128 - text embedding module that has same vocabulary and network as nnlm-en-dim128, but the
#                               weights were just randomly initialized and never trained on real data.
# And by training in two modes:
#
#     training only the classifier (i.e. freezing the module), and
#     training the classifier together with the module.

def train_and_evaluate_with_module(hub_module, train_module=False):
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence", module_spec=hub_module, trainable=train_module)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    estimator.train(input_fn=train_input_fn, steps=1000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    training_set_accuracy = train_eval_result["accuracy"]
    test_set_accuracy = test_eval_result["accuracy"]

    return {
        "Training accuracy": training_set_accuracy,
        "Test accuracy": test_set_accuracy
    }


results = {}
results["nnlm-en-dim128"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/nnlm-en-dim128/1")
results["nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/nnlm-en-dim128/1", True)
results["random-nnlm-en-dim128"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/random-nnlm-en-dim128/1")
results["random-nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
    "https://tfhub.dev/google/random-nnlm-en-dim128/1", True)

pd.DataFrame.from_dict(results, orient="index")

# Establish the baseline accuracy of the test set - the lower bound that can be achieved by outputting only
# the label of the most represented class:

estimator.evaluate(input_fn=predict_test_input_fn)["accuracy_baseline"]

# Assigning the most represented class will give us accuracy of 50%. There are a couple of things to notice here:
#
#     Maybe surprisingly, a model can still be learned on top of fixed, random embeddings. The reason is that even if
#           every word in the dictionary is mapped to a random vector, the estimator can separate the space purely
#           using its fully connected layers.
#     Allowing training of the module with random embeddings increases both training and test accuracy as oposed to
#           training just the classifier.
#     Training of the module with pre-trained embeddings also increases both accuracies. Note however the overfitting
#           on the training set. Training a pre-trained module can be dangerous even with regularization in the sense
#           that the embedding weights no longer represent the language model trained on diverse data, instead they
#           converge to the ideal representation of the new dataset.

