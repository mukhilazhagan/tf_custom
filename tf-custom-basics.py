# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


print(tf.add(1, 2))
print(tf.add([1,2], [3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.square(2) + tf.square(3)) ## Op Overloading

x = tf.matmul([[1]],[[2,3]]) # Matrix have double[[]]
print(x)
print(x.shape)
print(x.dtype)

# each tf.Tensor has a shape and dtype
# Tensors are immutable and can use GPU/TPU


# Compatability with Numpy
# TensorFlow operations automatically convert NumPy ndarrays to Tensors and vice versa
# .numpy() method for explicit conversion

import numpy as np

ndarray = np.ones([3,3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())


## Tensorflow and GPU
x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))



# TF Datasets
# tf.data.Datasets API to build a pipeline for feeding data to model

# Creating a source dataset using functions like
# Dataset.from_tensors
# Dataset.from_tensor_slices
# or using  TextLineDataset or TFRecordDataset.

ds_tensors = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6])
print(" Tensors from tensor slices")
print(ds_tensors)

# create a csv
import tempfile

_, filename = tempfile.mkstemp()

with open(filename, 'w') as f:
    f.write("""Line 1
Line 2
Line 3
""")

ds_file = tf.data.TextLineDataset(filename)
print(" Tensors from Text Line")
print(ds_file)


print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)

## Transformations are possible
# Use the transformations functions like map, batch,
# and shuffle to apply transformations to dataset records.

ds_tensors = ds_tensors.map(tf.square).shuffle(2).batch(2)

ds_file = ds_file.batch(2)


print('Elements of ds_tensors:')
for x in ds_tensors:
  print(x)

print('\nElements in ds_file:')
for x in ds_file:
  print(x)
















