import gym
import random
import numpy as np


LR = 1e-3
env = gym.make('CartPole-v0')
goal_steps = 500
basic_score_requirement = 50
initial_games = 10000
env.reset()


def random_games():
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			env.render()
			# generate and perform random actions
			action = env.action_space.sample()
			print(episode, t, action)
			observation, reward, done, info = env.step(action)
			print(observation, reward, done, info)
			if done:
				break

#random_games()
#env.reset()

def create_inital_population():
	training_data = []
	scores = []
	accepted_scores = []
	output = []
	for _ in range(initial_games):
		score = 0
		game_memory = []
		previous_obs = []
		for _ in range(goal_steps):
			action = random.randrange(0,2)
			observation, reward, done, info = env.step(action)

			if(len(previous_obs) > 0):
				game_memory.append([previous_obs, action])
			previous_obs = observation
			score += reward
			if done:
				break

		if score >= basic_score_requirement:
			accepted_scores.append([score])
			for data in game_memory:
				if data[1] == 1:
					output == [0, 1]
				elif data[1] == 0:
					output == [1, 0]

				training_data.append([data[0], output])
		env.reset()
		scores.append(score)

	#training_data = np.array(training_data)
	print('Average score in training_data:', np.mean(accepted_scores))
	print('Number of games in training_data:', len(accepted_scores))
	return training_data

#create_inital_population()

def neural_network_model(input_size):
	from collections import Counter
	import tflearn
	from tflearn.layers.core import input_data, fully_connected, dropout
	from tflearn.layers.estimator import regression

	network = input_data(shape = [None, input_size, 1], name = 'input')
	
	network = fully_connected(network, 128, activation = 'relu')
	network = dropout(network, 0.8)

	network = fully_connected(network, 256, activation = 'relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 512, activation = 'relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 256, activation = 'relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 128, activation = 'relu')
	network = dropout(network, 0.8)
	
	network = fully_connected(network, 2, activation = 'softmax')

	network = regression(network, optimizer = 'adam', learning_rate = LR, loss = 'categorical_crossentropy', name = 'targets')

	model = tflearn.DNN(network)
	return model

def train_model(training_data, model = False):
	from collections import Counter
	import tflearn
	from tflearn.layers.core import input_data, fully_connected, dropout
	from tflearn.layers.estimator import regression
	print(training_data[0])
	print(training_data[1])

	X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
	y = [i[1] for i in training_data]

	if not model:
		model = neural_network_model(input_size = len(X[0]))

	model.fit({'input': X}, {'targets': y}, n_epoch = 5, snapshot_step = 500, show_metric = True, run_id = 'openai_cartpole')
	return model

training_data = create_inital_population()
model = train_model(training_data)


scores = []
choices = []

for each_game in range(10):
	score = 0
	game_memory = [] 
	previous_obs = []
	env.reset()
	for _ in range(goal_steps):
		env.render()
		if len(previous_obs) == 0:
			action = random.randrange(0, 2)
		else:
			action = np.argmax(model.predict(previous_obs.reshape(-1, len(previous_obs), 1))[0])
		choices.append(action)
		new_observation, reward, done, info = env.step(action)
		previous_obs = new_observation
		game_memory.append(new_observation)
		score += reward
		if done:
			break
		scores.append(score)
		print('Average score in test_data:', np.mean(scores))
		print('Number of games in test_data:', len(scores))
