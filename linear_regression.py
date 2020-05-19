# make predictions

import numpy as np
import pandas as pd

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#display linear regression data visualization
import matplotlib.pyplot as plt 

# Load dataset
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# names = ['date', 'cases', 'deaths']

# read in data f
dataset = read_csv(url)

# set up data
# features
x = dataset['cases']
# target variable
y = dataset['deaths']

print("")
print("dataset.shape:")
print (dataset.shape)
print("")
print("dataset.describe:")
print(dataset.describe())
print("")


# Convert date to numerical value for use in Regression
#import datetime as dt
#x['Date'] = pd.to_datetime(x['Date'])
#x['Date'] = x['Date'].map(dt.datetime.toordinal)

# Split up data into train and validation/test
# First 80% of data assigned to training, 20% for testing, adjust with test_size
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

# Training the Model
x_train = np.array(x_train)
y_train = np.array(y_train)
x_validation = np.array(x_validation)
y_validation = np.array(y_validation)

print("")
print("x_train:")
print (x_train.shape)

# Reshape data for 
x_train_copy = x_train
x_train = x_train.reshape(-1,1)
x_validation = x_validation.reshape(-1,1)
y_train = y_train.reshape(-1,1)
y_validation = y_validation.reshape(-1,1)

print("")
print("x_train reshaped shape:")
print (x_train.shape)
print("")
print("x_trainshaped.describe:")
print(x_train.describe())
print("")


# Make predictions on validation dataset
#model = SVC(gamma='auto')
#model.fit(x_train, y_train)
#predictions = model.predict(x_validation)

# Evaluate predictions
#print(accuracy_score(y_validation, predictions))
#print(confusion_matrix(y_validation, predictions))
#print(classification_report(y_validation, predictions))

from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)

# make predictions on the test data
y_pred = clf.predict(x_validation)

#df = pd.DataFrame({'Actual': y_train.flatten(), 'Predicted': y_pred.flatten()})
#df

# Assess how good model is by comparing Predictions (y_pred) to real target values for test set (y_validation)
# calculate and display r2 score for model accuracy
print('R2 Score:',r2_score(y_validation,y_pred))
print("")

y_plot = []
#for i in range(10):
#   y_plot.append(y_pred)
plt.figure(figsize=(6,6))
plt.scatter(x_validation,y_validation,color='red',label='REDSCATTER')
plt.plot(x_train, y_pred, color='black', label='PLOTBLK')
#plt.plot(range(len(y_plot)),y_plot,color='black',label = 'BLACK')
plt.legend()
plt.show()