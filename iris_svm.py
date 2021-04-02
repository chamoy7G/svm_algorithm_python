import pandas as pd
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# Read in data
iris = load_iris()


# Distinguish between training sets and test sets
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3)

# SVM
model=SVC(C=10.0)
model.fit(X_train,y_train)

# Predict
print(str(model.predict(X_test)))
print(y_test)
# Accuracy
print('Accuracy: '+str(model.score(X_test,y_test)
))