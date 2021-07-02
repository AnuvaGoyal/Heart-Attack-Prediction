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
