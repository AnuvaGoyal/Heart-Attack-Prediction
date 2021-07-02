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
