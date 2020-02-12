#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# Use tf.Variable to represent weights in a model.

v = tf.Variable(1.0)
# Using python's assert as a debugging statement to test the condition

print(v == 1)
# vs
assert v.numpy() == 1.0, "Tf variable not True"

# To rename v :
# v = 2 doesn't work
v.assign(3.0)
assert v.numpy() == 3.0

# Reassigning

v.assign(tf.square(v))
assert v.numpy() == 9.0

# Using Tensor, Variable, and GradientTape—to build and train a simple model.

#1-Define the model.
#2-Define a loss function.
#3-Obtain training data.
#4-Run through the training data and use an "optimizer" to adjust the variables to fit the data.

# Simple Linear Model f(x) = x * W + b

class Model(object):
  def __init__(self):
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.W * x + self.b

model = Model()

assert model(3.0).numpy() == 15.0

# Using standard L2 loss, also known as the least square errors:

def loss(predicted_y, target_y):
  return tf.reduce_mean(tf.square(predicted_y - target_y))


# Synth Input - Adding Random Noise
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

import matplotlib.pyplot as plt

plt.scatter(inputs, outputs, c='b')
plt.scatter(inputs, model(inputs), c='r')
plt.show()

print('Current loss: %1.6f' % loss(model(inputs), outputs).numpy())

# There are many optimizers in tf.train.Optimizer, but we will implement it ourselves

# Implementing Gradient Descent

def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(model(inputs), outputs)
  dW, db = t.gradient(current_loss, [model.W, model.b])
  model.W.assign_sub(learning_rate * dW) # Assign by subtracting from it
  model.b.assign_sub(learning_rate * db)


model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
  Ws.append(model.W.numpy())  # First epoch, has __init__ weights
  bs.append(model.b.numpy())  # Similarly __init__ bias
  current_loss = loss(model(inputs), outputs)  # Output it is noise induced data

  train(model, inputs, outputs, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss)) # The most recent element to be printed

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'True W', 'True b'])
plt.show()





















