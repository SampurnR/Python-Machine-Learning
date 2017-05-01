# basic operations with constants
import tensorflow as tf

a = tf.constant(3)
b = tf.constant(5)

with tf.Session() as sess:
    print "Addition with constants: %i" % sess.run(a+b)
    print "Multiplication with constants: %i " % sess.run(a*b)


# basic operations with variables
import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print "Addition with variables %i" % sess.run(add, feed_dict = {a : 3, b : 5})
    print "Multiplication with variables %i" % sess.run(mul, feed_dict = {a : 3, b : 5})

# basic operations with matrices
import tensorflow as tf

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[5.], [5.]])

mul = tf.matmul(matrix1, matrix2)

with tf.Session() as sess:
    print "Matrix multiplication %i" % sess.run(mul)