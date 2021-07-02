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
