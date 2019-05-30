from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
import numpy as np
import os


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1,b'Iris-virginica':2}
    return it[s]

path = r'C:\Users\User\Desktop\iris.data'
data = np.loadtxt(path,dtype=float, delimiter=',',converters={4: iris_type})

# print(data)

x,y = np.split(data,(4,),axis=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,train_size=0.6)

clf = svm.SVC(C=0.8, kernel = 'rbf', gamma = 20, decision_function_shape='ovr')
# print(y_train)
# # y_train的转置
# print(y_train.ravel())

clf.fit(x_train,y_train.ravel())
y_pre = clf.predict(x_test)
acc = metrics.accuracy_score(y_pre,y_test)
print(acc)
