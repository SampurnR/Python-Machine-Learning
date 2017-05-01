# import mnist library from tensorflow library
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

# load data 
X_train = mnist.train.images
Y_train = mnist.train.labels

X_test = mnist.test.images
Y_test = mnist.test.labels

batch_X, batch_Y = mnist.train.next_batch(64)