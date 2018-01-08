# Lab 10 MNIST and NN
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777)  # reproducibility

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['Age'].fillna(train['Age'].median(), inplace=True )


x_data = train[ ["Age","Pclass","Fare"] ].values
y_data = train[ ["Survived"] ].values

test['Age'].fillna(test['Age'].median(), inplace=True )


x_test_data = test[ ["Age","Pclass","Fare"] ].values

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost: ", cost_val)


print(sess.run([cost, hypothesis, train], feed_dict={X: x_test_data[5]}))

