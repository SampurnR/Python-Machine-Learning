# nearest neighbors
import numpy as np
import tensorflow as tf

## import mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

## creating training data
X_tr, Y_tr = mnist.train.next_batch(5000)
## creating test data
X_te, Y_te = mnist.test.next_batch(200)

## creating placeholders
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", 784)

## L1 distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices = 1)

pred = tf.arg_min(distance, 0)
accuracy  = 0.

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(len(X_te)):
        nn_index = sess.run(pred, feed_dict = {xtr : X_tr, xte : X_te[i, :]})
        print "Test ", i, "Prediction: ", np.argmax(Y_tr[nn_index]), "True Class: ", np.argmax(Y_te[i])
        if np.argmax(Y_tr[nn_index]) == np.argmax(Y_te[i]):
            accuracy += 1./len(X_te)
    print "Accuracy: ", accuracy