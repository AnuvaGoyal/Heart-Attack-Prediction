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
