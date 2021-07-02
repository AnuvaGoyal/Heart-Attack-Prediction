"""Heart_prediction.ipynb

IMPORTING LIBRARIES
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""IMPORTING DATASET"""

dataset = pd.read_csv("heart.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

"""SPLITTING THE DATASET"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 0)

print(X_train)

print(X_test)

print(y_train)

print(y_test)

"""FEATURE SCALING"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)

print(X_test)
