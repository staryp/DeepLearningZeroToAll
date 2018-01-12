import cv2
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import requests
import os
import shutil
import urllib
import re
import cuter
import tensorflow as tf
import numpy as np
import copy

targetDir = 'C:/work/staryp/DeepLearningZeroToAll/marketit/input'
label_csv = pd.read_csv("C:/work/staryp/DeepLearningZeroToAll/marketit/apperance_score.csv")

batch_size = 15


train_filenames = []
train_labels = []

test_filenames = []
test_labels = []

def read_images_from_disk(input_queue):
    ### queue[0] = name of images / queue[1] = labels of images
    label = input_queue[1]
    ### queue의 내용을 읽어서 image의 내용을 불러온 뒤, jpeg형식으로 디코딩.
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    ### 원래 이미지의 모양으로 만들어줌. 
    example.set_shape([300, 300, 3])
    return example, label

for dirName, subdirList, fileList in os.walk(targetDir):
    
    idx = 0
    for fname in fileList:

        if re.search('\.jpg$', fname) :

            imgPath = '%s/%s' % (dirName,fname)
            x,b =fname.split('.')
            id,_=x.split('_')
            d =  label_csv.loc[label_csv['id'] == float(id)]
            newrow = d['value'].values
            arr = [0 for _ in range(5)]
            
            print(newrow)
            if not newrow:
                arr[int(2)] = 1
            elif newrow[0].count('하') > 0:
                arr[int(0)] = 1
            elif newrow[0].count('중') > 0:
                arr[int(1)] = 1
            elif newrow[0].count('중상') > 0:
                arr[int(2)] = 1
            elif newrow[0].count('상') > 0:    
                arr[int(3)] = 1
            elif newrow[0].count('최상') > 0:
                arr[int(4)] = 1
            else:
                arr[int(2)] = 1
    
            if idx < 800:
                train_filenames.append(imgPath)
                train_labels.append(arr)
            else:
                test_filenames.append(imgPath)
                test_labels.append(arr)


            print(idx)
            idx = idx+1


#train data
train_images = tf.convert_to_tensor(train_filenames, dtype = tf.string)
train_img_labels = tf.convert_to_tensor(train_labels, dtype = tf.int32)

train_input_queue = tf.train.slice_input_producer([train_images, train_img_labels], num_epochs=None, shuffle=True)
train_image_list, train_label_list = read_images_from_disk(train_input_queue)

train_image_batch = tf.train.batch([train_image_list, train_label_list], batch_size=batch_size)

#test data
test_images = tf.convert_to_tensor(test_filenames, dtype = tf.string)
test_img_labels = tf.convert_to_tensor(test_labels, dtype = tf.int32)

test_input_queue = tf.train.slice_input_producer([test_images, test_img_labels], num_epochs=None, shuffle=True)
test_image_list, test_label_list = read_images_from_disk(test_input_queue)

test_image_batch = tf.train.batch([test_image_list, test_label_list], batch_size=batch_size)




# hyper parameters
learning_rate = 0.001
training_epochs = 15

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 270000])
X_img = tf.reshape(X, [-1, 300, 300, 3])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 5])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)


# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 128 * 4 * 4])

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5


# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.device('/device:GPU:0'):
    # initialize
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # train my model
    print('Learning stared. It takes sometime.')

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    c, _, image_tensor = sess.run([cost, optimizer,train_image_batch], feed_dict=train_image_batch) # image_batch tensor를 session에서 돌린다.

    for i in range(batch_size):
        plt.imshow(image_tensor[0][i])
        print (image_tensor[1][i])
        plt.show()

    coord.request_stop()
    coord.join(threads)


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