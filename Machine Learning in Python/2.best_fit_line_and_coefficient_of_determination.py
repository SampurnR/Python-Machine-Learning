import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style
import random
%matplotlib inline

style.use('fivethirtyeight')

def mean(numbers):
	return float(sum(numbers)) / max(len(numbers), 1)

def create_dataset(quantity, variance, step = 2, correlation = False):
	val = 1
	ys = []
	for i in range(quantity):
		y = val + random.randrange(-1 * variance, variance)
		ys.append(y)
		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -= step
	xs = [i for i in range(len(ys))]
	return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

def best_fit_line_and_intercept(xs, ys):
	m = ( (mean(xs) * mean(ys)) - mean(xs * ys) ) / ( (mean(xs) * mean(xs)) - mean(xs * xs) )
	b = mean(ys) - m * mean(xs)
	return m, b

def squared_error(ys_orig, ys_line):
	return sum((ys_orig - ys_line) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_reg_line = squared_error(ys_orig, ys_line)
	squared_error_y_mean  = squared_error(ys_orig, y_mean_line)
	return 1 - (squared_error_reg_line / squared_error_y_mean)


#xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
#ys = np.array([5, 7, 11, 13, 17, 19], dtype = np.float64)
# play with these
xs, ys = create_dataset(40, 40, 2, 'pos')

m, b = best_fit_line_and_intercept(xs, ys)

regression_line = [(m * x + b) for x in xs]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plot.scatter(xs, ys)
plot.plot(xs, regression_line)
plot.show