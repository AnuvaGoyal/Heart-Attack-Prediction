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
