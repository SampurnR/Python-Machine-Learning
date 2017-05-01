import numpy as np
from math import sqrt
import warnings
from collections import Counter
import pandas as pd
import random
from __future__ import division


def k_nearest_neighbors(data, predict, k = 3):
	if len(data) >= k:
		warnings.warn("K is set to a value less than the total voting groups.")
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
			distances.append([euclidean_distance, group])

	votes = [i[1] for i in sorted(distances)[:k]]
	#print(Counter(votes).most_common(1))
	vote_result = Counter(votes).most_common(1)[0][0]
	confidence = Counter(votes).most_common(1)[0][1]/k

	return vote_result, confidence

#dataset = {'k' : [[1, 2], [2, 3], [3, 1]], 'r' : [[6, 5], [7, 7], [8, 6]]}
#new_point = [5, 7]
#result = k_nearest_neighbors(dataset, new_point, 3)
#print(result)

df = pd.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace = True)
df.drop(['id'], 1, inplace = True)
df = df.astype(float).values.tolist()

random.shuffle(df)

test_size = 0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}

train_data = df[:-int(test_size * len(df))]
test_data = df[int(test_size * len(df)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])

for i in test_data:
	test_set[i[-1]].append(i[:-1])


correct = 0
total = 0

for group in test_set:
	for data in test_set[group]:
		vote, confidence = k_nearest_neighbors(train_set, data, 5)
		if group == vote:
			correct += 1
		total += 1

print(correct, total)
print('Accuracy ', correct/total)