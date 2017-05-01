# multilayer perceptron

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

## parameters
learning_rate = 0.001
training_epochs = 100
batch_size = 128

## network parameters
n_hl_1 = 256
n_hl_2 = 256
n_input = 784
n_classes = 10

X = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

## weights 
weights = {
    'h1' : tf.Variable(tf.random_normal([n_input, n_hl_1])),
    'h2' : tf.Variable(tf.random_normal([n_hl_1, n_hl_2])),
    'out' : tf.Variable(tf.random_normal([n_hl_2, n_classes]))
}

## biases
biases = {
    'b1' : tf.Variable(tf.random_normal([n_hl_1])),
    'b2' : tf.Variable(tf.random_normal([n_hl_2])),
    'out' : tf.Variable(tf.random_normal([n_classes]))
}

## MLP
def multilayer_perceptron(x, weights, biases):
    hl1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hl1 = tf.nn.relu(hl1)
    
    hl2 = tf.add(tf.matmul(hl1, weights['h2']), biases['b2'])
    hl2 = tf.nn.relu(hl2)
    
    out_layer = tf.matmul(hl2, weights['out']) + biases['out']
    return out_layer

## model
pred = multilayer_perceptron(X, weights, biases)

## cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
## optimizer
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size) 
            _, c = sess.run([optimizer, cost], feed_dict = {X : batch_x, y: batch_y})
            avg_cost += c/total_batch
        print "Epoch: ", epoch, "Cost: ", c
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print "Accuracy: ", accuracy.eval({X : mnist.test.images, y : mnist.test.labels})