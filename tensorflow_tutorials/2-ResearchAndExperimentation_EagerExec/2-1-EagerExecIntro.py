
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/eager/eager_basics.ipynb

import tensorflow as tf

tf.enable_eager_execution() # "A more interactive mode'

print(tf.add(1, 2))
# tf.Tensor(3, shape=(), dtype=int32)

print(tf.add([1, 2], [3, 4]))
# tf.Tensor([4 6], shape=(2,), dtype=int32)

print(tf.square(5))
# tf.Tensor(25, shape=(), dtype=int32)

print(tf.reduce_sum([1, 2, 3]))
# tf.Tensor(6, shape=(), dtype=int32)

print(tf.encode_base64("hello world"))
# tf.Tensor(b'aGVsbG8gd29ybGQ', shape=(), dtype=string)

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))
# tf.Tensor(13, shape=(), dtype=int32)


x = tf.matmul([[1]], [[2, 3]]) # x = tf.Tensor([[2 3]], shape=(1, 2), dtype=int32)
print(x.shape) # (1, 2)
print(x.dtype) # (int32)


# ## NumPy Compatibility¶

# Conversion between TensorFlow Tensors and NumPy ndarrays is quite simple as:
#
#     TensorFlow operations automatically convert NumPy ndarrays to Tensors.
#     NumPy operations automatically convert Tensors to NumPy ndarrays.


import numpy as np

ndarray = np.ones([3, 3])
# TensorFlow operations convert numpy arrays to Tensors automatically"

tensor = tf.multiply(ndarray, 42)
print(tensor)
# tf.Tensor(
# [[42. 42. 42.]
#  [42. 42. 42.]
#  [42. 42. 42.]], shape=(3, 3), dtype=float64)


# And NumPy operations convert Tensors to numpy arrays automatically"
print(np.add(tensor, 1))

# "The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

# ## GPU acceleration

x = tf.random_uniform([3, 3])
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0')) # The Tensor.device property provides a fully qualified string name of the
                                    # device hosting the contents of the Tensor.

# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    tf.matmul(x, x)
    print("done")

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        tf.matmul(x, x)
        print("done")


# ## Datasets
# This section demonstrates the use of the tf.data.Dataset API to build pipelines to feed data to your model.
# ### Create a source dataset

ds_tensors = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5, 6])

# Create a CSV file
import tempfile
_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3
  """)

ds_file = tf.data.TextLineDataset(filename)


# ### Apply transformations
# Use the transformations functions like map, batch, shuffle etc. to apply transformations to the records of the dataset.

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)
ds_file = ds_file.batch(2)

# ### Iterate


print('Elements of ds_tensors:')
for x in ds_tensors:
    print(x)

print('\nElements in ds_file:')
for x in ds_file:
    print(x)


