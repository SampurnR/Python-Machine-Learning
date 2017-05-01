import pandas as pd
import quandl

df = quandl.get("BSE/BOM539254")

df = df[['Open', 'High', 'Low', 'Close', 'No. of Trades']]
df['hl_prcnt'] = ((df['High'] - df['Close'])/df['Close']) * 100
df['change_prcnt'] = ((df['Close'] - df['Open'])/df['Open']) * 100

df = df[['Close', 'hl_prcnt', 'change_prcnt', 'No. of Trades']]
print(df.head())

###################################################

# creating training dataset, and prediction column
forecast_col = 'Close'
df.fillna(-99999, inplace = True)

import math
# using last 41 (forecast_out) days to predict current day's closing value.
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)
# shifts up by 41 (forecast_out) rows and saves to predicted column
df['predicted'] = df[forecast_col].shift(-forecast_out)

###################################################

import numpy as np
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression

# creating features
df.dropna(inplace = True)
x = np.array(df.drop(['predicted'], axis = 1))
x = preprocessing.scale(x)

# creating labels
y = np.array(df['predicted'])

# creating training and test data
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

# creating a classifier
classifier = LinearRegression()

# training the classifier model
classifier.fit(X_train, Y_train)

# testing the classifier model
accuracy = classifier.score(X_test, Y_test)
print(accuracy)


###################################################

# creating features
x = np.array(df.drop(['predicted'], axis = 1))
x = preprocessing.scale(x)
x_latest = x[:-forecast_out]
x = x[:-forecast_out]

df.dropna(inplace = True)

# creating labels
y = np.array(df['predicted'])

# creating training and test data
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

# creating a classifier
classifier = LinearRegression()

# training the classifier model
classifier.fit(X_train, Y_train)

# testing the classifier model
accuracy = classifier.score(X_test, Y_test)
print(accuracy)

predicted_values = classifier.predict(x_latest)

###################################################

