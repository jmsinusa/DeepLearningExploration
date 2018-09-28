# Try to work out why CNNs generate CuDNN error.

import numpy as np
import tensorflow as tf

# Get MNIST

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)



def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) # -1 is effectively a wildcard
    pool1 = tf.layers.max_pooling2d(inputs=input_layer, pool_size=[14, 14], strides=14)
    pool1_flat = tf.reshape(pool1, [-1, 4])
    dense = tf.layers.dense(inputs=pool1_flat, units=10, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    # We name this layer, so we can refer to it later.
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # We compile our predictions in a dict, and return an EstimatorSpec object:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)  # This includes the softmax

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# setup train_input_fn

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_train},
    y=y_train,
    batch_size=100,
    num_epochs=None,
    shuffle=True)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": x_test},
    y=y_test,
    num_epochs=1,
    shuffle=False)

# Set up estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

mnist_classifier.train(
    input_fn=train_input_fn,
    steps=1000)
# mnist_classifier.train(
#     input_fn=train_input_fn,
#     steps=20000,
#     hooks=[logging_hook])


print ('DONE')