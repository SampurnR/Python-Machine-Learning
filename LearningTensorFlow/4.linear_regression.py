# linear regression
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random

## parametes
learning_rate = 0.001
training_epochs = 1000
display_step = 50

## training data
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

## placeholders
X = tf.placeholder("float")
Y = tf.placeholder("float")

## coefficients for the linear equation
W = tf.Variable(rng.random(), name = "weight")
b = tf.Variable(rng.random(), name = "bias")

## linear model
pred = tf.add(tf.multiply(X, W), b)
## mean square error
cost = tf.reduce_sum(tf.pow(pred - Y, 2))/(2 * n_samples)

## gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict = {X : x, Y : y})
            
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict = {X : x, Y : y})
            print "Epoch: ", (epoch + 1), "cost: ", c, "weight: ", sess.run(W), "bias: ", sess.run(b)
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()