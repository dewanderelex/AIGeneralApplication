# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:01:47 2018

@author: Con_0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
with open("winequality-white.csv") as f:
    reader = csv.reader(f)
    next(reader) # skip header
    data = [r for r in reader]
y = []
for i in range(len(data)):
    data[i] = data[i][0].split(";")

for i in range (len(data)):
    for j in range (len(data[i])):
        data[i][j] = float(data[i][j])
        if j == len(data[i]) - 1:
            y.append(data[i][j])
            del data[i][j]
X = np.array(data)

y = np.array(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressorr = LinearRegression()
regressorr.fit(X_train, y_train)
y_predict = regressorr.predict(X_test)

X_firstFeature = []
for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        if j == 0:
            X_firstFeature.append(X_test[i][j])

plt.plot(np.array(X_firstFeature), y_test, label = "Actual Data")
plt.plot(np.array(X_firstFeature), y_predict, label = "Predicted Data")
plt.title("Wine quality based on features")
plt.legend()
plt.show()
