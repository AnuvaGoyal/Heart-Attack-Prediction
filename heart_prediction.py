# -*- coding: utf-8 -*-
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


"""PLOTS

"""

dataset.hist(figsize=(16,10))
plt.show()

import seaborn as sns
plt.figure(figsize=(10,10))
sns.heatmap(dataset.corr(), annot=True)

sns.countplot(dataset['output'])
plt.show()

sns.displot(x= 'age', hue='output', data=dataset, alpha=0.6)
plt.show()

attack = dataset[dataset['output']==1]
sns.displot(attack.age, kind = 'kde')
plt.show()

sns.displot(attack.age, kind = 'ecdf')
plt.grid(True)
plt.show()

ranges = [0, 30, 40, 50, 60, 70, np.inf]
labels = ['0-30', '30-40', '40-50', '50-60', '60-70', '70+']

attack['age'] = pd.cut(attack['age'], bins=ranges, labels=labels)
attack['age'].head()


sns.countplot(attack.age)
plt.show()

fig, ax= plt.subplots(figsize=(8, 5))
sns.countplot(x= 'sex', hue='age', data=attack, ax=ax)
ax.set_xticklabels(['Female', 'Male'])
plt.legend(loc = 'upper right')
plt.show()


"""LOGISTIC REGRESSION"""

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


"""PREDICTING"""

y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

print(classifier.predict(sc.transform([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])))


"""CONFUSION MATRIX"""

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


"""TRAINING AND TEST SCORE"""

print("Training score:{:.3f}".format(classifier.score(X_train, y_train)))
print("Test score:{:.3f}".format(classifier.score(X_test, y_test)))


"""RANDOM FOREST CLASSIFICATION"""

from sklearn.ensemble import RandomForestClassifier
classifier2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0, max_depth=3)
classifier2.fit(X_train, y_train)

y_pred2 = classifier2.predict(X_test)
print(np.concatenate((y_pred2.reshape(len(y_pred2),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm2 = confusion_matrix(y_test, y_pred2)
print(cm2)
accuracy_score(y_test, y_pred2)

print("Training score:{:.3f}".format(classifier2.score(X_train, y_train)))
print("Test score:{:.3f}".format(classifier2.score(X_test, y_test)))


"""K-N NEIGHBORS

"""

from sklearn.neighbors import KNeighborsClassifier
classifier3=KNeighborsClassifier(n_neighbors=4, metric='minkowski', p=2)
classifier3.fit(X_train, y_train)

y_pred3 = classifier3.predict(X_test)
print(np.concatenate((y_pred3.reshape(len(y_pred3),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm3 = confusion_matrix(y_test, y_pred3)
print(cm3)
accuracy_score(y_test, y_pred3)

print("Training score:{:.3f}".format(classifier3.score(X_train, y_train)))
print("Test score:{:.3f}".format(classifier3.score(X_test, y_test)))


"""KERNEL SVM"""

from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf', random_state = 0)
classifier4.fit(X_train, y_train)

y_pred4 = classifier.predict(X_test)
print(np.concatenate((y_pred4.reshape(len(y_pred4),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm4 = confusion_matrix(y_test, y_pred4)
print(cm4)
accuracy_score(y_test, y_pred4)

print("Training score:{:.3f}".format(classifier4.score(X_train, y_train)))
print("Test score:{:.3f}".format(classifier4.score(X_test, y_test)))


"""NAIVE BAYES"""

from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, y_train)

y_pred5 = classifier5.predict(X_test)
print(np.concatenate((y_pred5.reshape(len(y_pred5),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm5 = confusion_matrix(y_test, y_pred5)
print(cm5)
accuracy_score(y_test, y_pred5)

print("Training score:{:.3f}".format(classifier5.score(X_train, y_train)))
print("Test score:{:.3f}".format(classifier5.score(X_test, y_test)))


"""DECISION TREE CLASSIFIER"""

from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion = 'gini', random_state = 0, max_depth=3)
classifier6.fit(X_train, y_train)

y_pred6 = classifier6.predict(X_test)
print(np.concatenate((y_pred6.reshape(len(y_pred6),1), y_test.reshape(len(y_test),1)),1))

from sklearn.metrics import confusion_matrix, accuracy_score
cm6 = confusion_matrix(y_test, y_pred)
print(cm6)
accuracy_score(y_test, y_pred)

print("Training score:{:.3f}".format(classifier6.score(X_train, y_train)))
print("Test score:{:.3f}".format(classifier6.score(X_test, y_test)))
