import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt

tf.set_random_seed(777)  # reproducibility

# load csv file  

filename = './201712242147.csv'
df = pd.read_csv(filename, index_col="id")
df.head()

# train Parameters

features = ['open', 'close', 'min', 'max', 'avg', 'trade', 'cur_buy', 'cur_sell']

# RNN cell param

seq_length = 20 
data_dim = len(features)
lstm_hidden_size = 8
output_dim = 1
learning_rate = 0.01
iterations = 500
precit_min_add = 1
lstm_layer_num = 3
dropout_ratio = 0.1
ffnn_width = 5

def MinMaxScaler(data):    
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

xy = df[features].values
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []

for i in range(0, len(y) - seq_length - precit_min_add ):
    _x = x[ i : i + seq_length ]
    _y = y[ i + seq_length + precit_min_add ]
    dataX.append(_x)
    dataY.append(_y)
                 

# train/test split

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

# input place holders

drop_out_ratio = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, output_dim])


# Make a lstm cell with hidden_size (each unit output vector size)
def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(lstm_hidden_size, state_is_tuple=True)
    return cell

multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lstm_layer_num)], state_is_tuple=True)

# outputs: unfolding size x hidden size, state = hidden size
outputs, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)

L1 = tf.contrib.layers.fully_connected(outputs[:, -1], ffnn_width, activation_fn=tf.nn.relu)  # We use the last cell's output
L1 = tf.contrib.layers.dropout(L1, keep_prob=drop_out_ratio)
L2 = tf.contrib.layers.fully_connected(L1, ffnn_width, activation_fn=tf.nn.relu)  # We use the last cell's output
L2 = tf.contrib.layers.dropout(L2, keep_prob=drop_out_ratio)
L3 = tf.contrib.layers.fully_connected(L2, ffnn_width, activation_fn=tf.nn.relu)  # We use the last cell's output
L3 = tf.contrib.layers.dropout(L3, keep_prob=drop_out_ratio)
Y_pred = tf.contrib.layers.fully_connected(L3, output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, output_dim])
predictions = tf.placeholder(tf.float32, [None, output_dim])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

with tf.device('/device:GPU:0'):
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        _, step_loss = sess.run([train, loss], feed_dict={
                                X: trainX, Y: trainY, drop_out_ratio:dropout_ratio})
        if i % 20 == 0:
            print("[step: {}] loss: {}".format(i, step_loss))
    
    test_predict = sess.run(Y_pred, feed_dict={X: testX, drop_out_ratio:1.0})
    rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

plt.figure(figsize=(15,10))
plt.plot([v for v in testY[0:500] ], label='real')
plt.plot([v for v in test_predict[0:500]], label='predict')
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()