# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 01:13:31 2018

@author: OM SAI RAM
"""
#IMPORTING LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING DATASET

dataset = pd.read_csv('50_Startups.csv')

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#ENCODING CATEGORICAL DATA
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:,1:]

#Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)