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

#Getting MNIST data
mnist = fetch_mldata('MNIST original')

#Dividing the data into Label and Target
X,y=mnist["data"],mnist["target"]

#Dividing the data into Training Dataset and Testing Dataset
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]


# Naive bayse
gnb = GaussianNB()

#Traing my model
gnb.fit(X_train, y_train)

#prediction
y_pred_sgd = gnb.predict(X_test)

#Accuracy
acc_gnb = accuracy_score(y_test, y_pred_sgd)

print ("Gaussian Naive Bayes accuracy: ",acc_gnb)


"""Sometimes You will get 99% Accuracy but you need to check the cross validation values 
and precision score as well as the recall score. """

cross_val=cross_val_score(gnb, X_train, y_train, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(gnb, X_train, y_train, cv=3)
print(y_train_pred)
conf_mx=confusion_matrix(y_train, y_train_pred)
print(conf_mx)
ps=precision_score(y_train, y_train_pred,average="macro")
print("precision score:",ps)
rs=recall_score(y_train, y_train_pred,average="macro")
print("Recall Score:",rs)

"""Here i am showing you the graph of the confussion matrix. more the bright more erroneous it is. 
    so by getting the confussion matrix graph we canlearn from it and we can change the values of 
    classifier or by removing the noise from the images we can get the optimized result"""

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
print("precision score:",ps)
rs=recall_score(y_train, y_train_pred,average="macro")
print("Recall Score:",rs)
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
print("precision score:",ps)
rs=recall_score(y_train, y_train_pred,average="macro")
print("Recall Score:",rs)
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
print("precision score:",ps)
rs=recall_score(y_train, y_train_pred,average="macro")
print("Recall Score:",rs)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
