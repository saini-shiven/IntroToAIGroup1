# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:51:59 2022

@author: zainab
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load phishing dataset into dataframe
phishing = pd.read_csv("phishingDataset.csv")


# Select all columns but price (which is the target) as data features
X = phishing[ phishing.columns[phishing.columns!='Result'] ] 
# Select price as target
y = phishing['Result']

#split into testing and training
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 

#For the following models, some experimentation has been performed
#in order to find the appropriate parameter for the models.
#This isn't given here, but you might try varying the values.
#You might well want to plot the curves that your find.

#set up kNN
knn_model = KNeighborsClassifier(n_neighbors=12)



#fit and test kNN
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_pred_knn))




# train the model
knn_model.fit(X_train,y_train)

# make predictions using the testing data
y_pred = knn_model.predict(X_test)




def evaluateAccuracy(predictions):
    # calculate accuracy of model
    accuracy = accuracy_score(y_test, predictions)
    # rounded to 2 significant figures
    print('Accuracy: %.3f' % accuracy)
    
    # produce confusion matrix
    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
    display.plot()
    plt.show()

evaluateAccuracy(y_pred)