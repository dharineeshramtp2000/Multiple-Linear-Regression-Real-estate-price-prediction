# import the libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
Dataset = pd.read_csv("Real estate.csv")
X = Dataset.iloc[:, 2:7]
y = Dataset.iloc[:, -1]

#Feature scaling the independent variable
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)

#importing the linear regression models(using SGD)
from sklearn.linear_model import SGDRegressor

# Creating the object instance for SGD(Stochastic Gradient Descent)
regressor = SGDRegressor(max_iter=10000, tol=1e-3, alpha =0.01, random_state = 0, learning_rate = 'invscaling' , eta0 = 0.0001)
regressor.fit(X_train, y_train)

#Predicting the output for our SGD Linear Model with the test set
y_pred2 = regressor.predict(X_test)

# Now lets calculate the Coefficient of Determination
from sklearn.metrics import r2_score, mean_squared_error

r_squared = r2_score(y_test, y_pred2)
print(r_squared)

