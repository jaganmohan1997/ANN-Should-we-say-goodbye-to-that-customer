# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 22:55:27 2019

@author: Jagan Mohan
"""

# Install Tensorflow & Keras Before Running this script

# Part1 - Data Pre Processing

#Imoprting the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the Dataset
df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:,3:-1].values
y = df.iloc[:,-1].values

#Checking for missing values
from sklearn.preprocessing import Imputer
X1 = df.iloc[:,3:-1]
X1.isnull().sum()
## Since there are no missing values we can continue without any Missing Value Imputation


#Encoding the Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
X[:,1] = labelencoder_1.fit_transform(X[:,1])
labelencoder_2 = LabelEncoder()
X[:,2] = labelencoder_2.fit_transform(X[:,2])
onhotencoder = OneHotEncoder(categorical_features = [1])
X = onhotencoder.fit_transform(X).toarray()

#Removing redundant variable due to One hot encoding
X = X[:, 1:]

# Splitting the data before running the ANN
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 1)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
standardscaler = StandardScaler()
X_train = standardscaler.fit_transform(X_train)
X_test = standardscaler.transform(X_test)

# Part 2 - Artificial Neural Network

#Importing keras libraries and packages
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

classifier = Sequential()

#Initializing the ANN with 2 hidden layers of 6 nodes each

classifier = Sequential([Dense(6, input_shape=(11,)), Activation('relu'),
                         Dense(6), Activation('relu'),
                         Dense(1), Activation('sigmoid')
                         ])

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size = 20, epochs = 100)


# Part 3 - Making Predictions and Evaluating the Model

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Let's make a confusion matrix and find out the accuracy on test dataset
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Part 4 - Now let us predict whether the following customer Churns out or not
X_oot = np.array([[600,'France','Male',40,3,60000,2,1,1,50000]], dtype = 'object')

X_oot[:,1] = labelencoder_1.transform(X_oot[:,1])
X_oot[:,2] = labelencoder_2.transform(X_oot[:,2])
X_oot = onhotencoder.transform(X_oot).toarray()
X_oot = X_oot[:, 1:]
X_oot = standardscaler.transform(X_oot)

y_oot = classifier.predict(X_oot)
y_oot = (y_oot > 0.5)
y_oot
## Hurrah!! The above Customer Will not Churnout and stays with the bank.












