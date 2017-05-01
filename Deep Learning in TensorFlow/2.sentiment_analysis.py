import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter
import tensorflow as tf

lemmatizer = WordNetLemmatizer()
line_nums = 100000


def create_lexicon(pos, neg):
	lexicon = []
	for fi in [pos, neg]:
		with open(fi, 'r') as f:
			contents = f.readlines()
			for l in contents[:line_nums]:
				all_words = word_tokenize(l.lower().decode('utf-8'))
				lexicon += list(all_words)

	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	w_count = Counter(lexicon)

	lexicon2 = []
	for w in w_count:
		if 1000 > w_count[w] > 50:
			lexicon2.append(w)

	print(len(lexicon2))
	return(lexicon2)

def sample_handling(sample, lexicon, classification):
	featureset = []

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:line_nums]:
			current_words = word_tokenize(l.lower().decode('utf-8'))
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word)
					features[index_value] += 1
				featureset.append(features)
	return featureset

def create_featuresets_and_labels(pos, neg, test_size = 0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	features = sample_handling(pos, lexicon, [1, 0])
	features += sample_handling(neg, lexicon, [0, 1])
	random.shuffle(features)

	features = np.array(features)

	testing_size = int(test_size * len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y


train_x, train_y, test_x, test_y = create_featuresets_and_labels('/home/sampurn/projects/Python & Machine Learning/Deep Learning in TensorFlow/data/pos.txt', '/home/sampurn/projects/Python & Machine Learning/Deep Learning in TensorFlow/data/neg.txt')
with open('sentiment_set.pickle', 'wb') as f:
	pickle.dump([train_x, train_y, test_x, test_y], f) 


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 2
batch_size = 100


x = tf.placeholder('float', [None, len(train_x[0])])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])), 
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 
					  'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 
					  'biases': tf.Variable(tf.random_normal([n_classes]))}

	# (input_data * weights) + biases

	l1 = tf.add(tf.matmul(data, hidden_layer_1['weights']), hidden_layer_1['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction, y) )
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	epochs_num = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(epochs_num):
			epoch_cost = 0
			i = 0
			while i < len(train_x):
				start = i
				end = i + batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])
				_, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
				epoch_cost += c
				i = batch_size
			print('Epoch', epoch, ' completed out of', epochs_num, 'epochs. Cost:', epoch_cost)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy', accuracy.eval({x: test_x, y: test_y}))

train_neural_network(x)