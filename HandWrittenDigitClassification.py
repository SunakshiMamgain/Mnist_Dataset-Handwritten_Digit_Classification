from sklearn.datasets import fetch_mldata
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import timeit
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import warnings
from sklearn.metrics import precision_score, recall_score
warnings.filterwarnings("ignore")

mnist = fetch_mldata('MNIST original')
X,y=mnist["data"],mnist["target"]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
"""
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_sgd = gnb.predict(X_test)
acc_gnb = accuracy_score(y_test, y_pred_sgd)

print ("Gaussian Naive Bayes accuracy: ",acc_gnb)

cross_val=cross_val_score(gnb, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(gnb, X_train, y_train, cv=3)
print(y_train_pred)
conf_mx=confusion_matrix(y_train, y_train_pred)
print(conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print(ps)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_sgd = clf.predict(X_test)
acc_clf = accuracy_score(y_test, y_pred_sgd)

print ("Decision Tree Classifier accuracy: ",acc_gnb)

cross_val=cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
print(y_train_pred)
conf_mx=confusion_matrix(y_train, y_train_pred)
print(conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print(ps)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)

clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
y_pred_rf = clf_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print ("random forest accuracy: ",acc_rf)
cross_val=cross_val_score(clf_rf, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf_rf, X_train, y_train, cv=3)
print(y_train_pred)
conf_mx=confusion_matrix(y_train, y_train_pred)
print(conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print(ps)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)


clf_svm = LinearSVC()
clf_svm.fit(X_train, y_train)
y_pred_svm = clf_svm.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print ("Linear SVM accuracy: ",acc_svm)
cross_val=cross_val_score(clf_svm, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf_svm, X_train, y_train, cv=3)
print(y_train_pred)
conf_mx=confusion_matrix(y_train, y_train_pred)
print(conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print(ps)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
"""
clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print ("nearest neighbors accuracy: ",acc_knn)
cross_val=cross_val_score(clf_knn, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf_knn, X_train, y_train, cv=3)
print(y_train_pred)
conf_mx=confusion_matrix(y_train, y_train_pred)
print(conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print(ps)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
