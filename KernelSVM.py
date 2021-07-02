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
