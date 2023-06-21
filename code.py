#importing all the required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

#reading the csv data file
dataset=pd.read_csv("diabetes.csv")
data=dataset.values
X=data[:,:-1]
Y=data[:,8:]

#splitting the dataset into train(75%) and test(25%) samples
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

#applying data cleaning i.e., preprocessing
scaler= StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#the knn classifier
classifier=KNeighborsClassifier(n_neighbors=10)
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
print("The KNN Classification details are as follows: \n")
print(classification_report(Y_test,Y_pred))
print("The Confusion Matrix for KNN:\n",confusion_matrix(Y_test,Y_pred))
a=accuracy_score(Y_test,Y_pred)
print("The Accuracy for KNN = ",a)

#the random forest classifier
classifier=RandomForestClassifier(n_estimators=50)
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
print("The Random Forest Classification details are as follows: \n")
print(classification_report(Y_test,Y_pred))
print("The Confusion Matrix for Random Forest:\n",confusion_matrix(Y_test,Y_pred))
b=accuracy_score(Y_test,Y_pred)
print("The Accuracy for Random Forest = ",b)

#the logistic regression classifier classifier
classifier=LogisticRegression()
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)
print("The Logistic Regression Classification details are as follows: \n")
print(classification_report(Y_test,Y_pred))
print("The Confusion Matrix for Logistic Regression:\n",confusion_matrix(Y_test,Y_pred))
c=accuracy_score(Y_test,Y_pred)
print("The Accuracy for Logistic Regression = ",c)

#comparing accuracies of all three methods
x=["KNN","Random Forest","Logistic Regression"]
y=[a,b,c]
plt.plot(x,y)
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('Comparing various Classifers')
plt.show()
