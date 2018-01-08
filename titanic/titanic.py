# Lab 10 MNIST and NN
import tensorflow as tf
import pandas as pd
import numpy as np
import random

tf.set_random_seed(777)  # reproducibility

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train['Age'].fillna(train['Age'].median(), inplace=True )
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

titanicX = train[ ["Age","Pclass","Fare","Sex"] ].values
titanicY = train[ ["Survived"] ].values

test['Age'].fillna(test['Age'].median(), inplace=True )
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

testX = test[ ["Age","Pclass","Fare","Sex"] ].values
testY = test[ ["Survived"] ].values


# parameters
learning_rate = 0.1
training_epochs = 15
batch_size = 10

# input place holders
X = tf.placeholder(tf.float32, [None, 4])
Y = tf.placeholder(tf.float32, [None, 1])

# weights & bias for nn layers
W1 = tf.Variable(tf.random_normal([4, 10]))
b1 = tf.Variable(tf.random_normal([10]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([10, 10]))
b2 = tf.Variable(tf.random_normal([10]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

W3 = tf.Variable(tf.random_normal([10, 1]))
b3 = tf.Variable(tf.random_normal([1]))
hypothesis = tf.matmul(L2, W3) + b3

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

def next_batch(data,s,l):
    return data[s:l]


# train my model
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(891/ batch_size)

    for i in range(total_batch):
        batch_xs = next_batch(titanicX, batch_size*i,batch_size*(i+1))
        batch_ys = next_batch(titanicY, batch_size*i,batch_size*(i+1))
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))



# for step in range(2001):
#     cost_val, hy_val, _ = sess.run([cost, optimizer],feed_dict={X: tatanicX, Y: tatanicY})
#     if step % 10 == 0:
#         print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

print("done")

