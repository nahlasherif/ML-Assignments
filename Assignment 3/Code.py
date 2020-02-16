import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
import os

#import the csv file and clean the data (remove empty rows that were imported)
data = pd.read_csv(os.getcwd() + "\Scripts\house_prices_data_training_data.csv")
data = data.dropna()

#save prices in y and remove from training data

y=data.price
datacopy = data
x=data.drop('price',axis=1)
x=data.drop('date',axis=1)
datacopy = datacopy.drop('date',axis=1)
datacopy = datacopy.drop('id', axis=1)
########################################################### ????

#split the data and y into training, validation and testing sections
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

#visualize number of bedrooms and price
print(X_train)

#pyplot.plot(noOfBedrooms, y_train, 'ro', ms=10, mec='k')
#pyplot.ylabel('Prices')
#pyplot.xlabel('Number of Bedrooms')
#pyplot.show()

#normalize the data
def  featureNormalize(X_train):
    mu = np.mean(X_train, axis=0);
    sigma = np.std(X_train, axis=0);
    X_norm = (X_train - mu) / sigma
    return X_norm

X_norm = featureNormalize(X_train)

#Features
m = np.size(X_train,0) #no of data recordings

#add intercept term to X
allones = np.ones(X_train.shape[0])
X_norm['x0'] = allones
print("X_normmm:")
print(X_norm)


def computeCost(X, y, theta):
    J = (np.dot(((np.dot(X, theta) - y).T), (np.dot(X, theta) - y))) / 2 * m
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
    J_history = []
    for i in range(num_iters):
        theta = theta - (alpha/m)*(np.dot(X,theta.T)-y).dot(X)
        J_history.append(computeCost(X, y, theta))
    return theta, J_history

# Choose some alpha value - change this
alpha = 0.01
num_iters = 1000
# init theta and run gradient descent
theta = np.zeros(np.size(X_norm,1)) #no. of features

#FOR FIRST HYPOTHESIS
theta1, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
pyplot.show()

# Display the gradient descent's result
print('theta1 computed from gradient descent: {:s}'.format(str(theta1)))

#FOR SECOND HYPOTHESIS
X_norm["bedrooms"] = np.square(X_norm["bedrooms"])
theta2, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
pyplot.show()

# Display the gradient descent's result
print('theta2 computed from gradient descent: {:s}'.format(str(theta2)))

#FOR THIRD HYPOTHESIS
X_norm["view"] = np.square(X_norm["view"])
theta3, J_history = gradientDescent(X_norm, y_train, theta, alpha, num_iters)

# Plot the convergence graph
pyplot.plot(np.arange(len(J_history)), J_history, lw=2)
pyplot.xlabel('Number of iterations')
pyplot.ylabel('Cost J')
pyplot.show()

# Display the gradient descent's result
print('theta3 computed from gradient descent: {:s}'.format(str(theta3)))


X_normval = featureNormalize(X_val)
allones = np.ones(X_val.shape[0])
X_normval ['x0'] = allones

print("Validation on thetas:")
#VALIDATION theta1
J1 = computeCost(X_normval, y_val, theta1)
print("J1",J1)

#VALIDATION theta2
X_normval["bedrooms"] = np.square(X_normval["bedrooms"])
J2 = computeCost(X_normval, y_val, theta2)
print("J2",J2)

#VALIDATION theta3
X_normval["view"] = np.square(X_normval["view"])
J3 = computeCost(X_normval, y_val, theta3)
print("J3",J3)

X_normtest = featureNormalize(X_test)
allones = np.ones(X_test.shape[0])
X_normtest ['x0'] = allones

print("Test on thetas:")

#TESTTT theta1
J4 = computeCost(X_normtest, y_test, theta)
print("J4",J4)

#TESTTT theta2
X_normtest["bedrooms"] = np.square(X_normtest["bedrooms"])
J5 = computeCost(X_normtest, y_test, theta2)
print("J5",J5)

#TESTTT theta3
X_normtest["view"] = np.square(X_normtest["view"])
J6 = computeCost(X_normtest, y_test, theta3)
print("J6",J6)

from sklearn.model_selection import KFold
#added some parameters
kf = KFold(n_splits = 3, shuffle = True, random_state = 2)
result = next(kf.split(datacopy), None)
print (result)

train = datacopy.iloc[result[0]] #Y included
test =  datacopy.iloc[result[1]]
ytrain=train.price
train =train.drop('price',axis=1)
ytest=test.price
test =test.drop('price',axis=1)


print (train)
print(test)
