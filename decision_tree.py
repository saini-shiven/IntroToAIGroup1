#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 10:23:35 2022

@author: andreas
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd

# load phishing dataset into dataframe
phishing = pd.read_csv("phishingDataset.csv")

# split the dataset into features (X) and targets (y)
X = phishing.drop(["id","Result"], axis=1)
y = phishing.Result

# split the dataset into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# WITH ENTROPY CRITERION

# create an instance of a decision tree classifer (split on entropy)
decisionTree = DecisionTreeClassifier(criterion='entropy')

# train the model
decisionTree.fit(X_train,y_train)

# make predictions using the testing data
y_pred = decisionTree.predict(X_test)

# function that uses metrics to check accuracy and plot confusion matrix
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


# WITH GINI CRITERION AND BEST SPLITTER

# create instance of decision tree
decisionTree2 = DecisionTreeClassifier(criterion='gini')
# train the model
decisionTree2.fit(X_train,y_train)

# make predictions using the testing data
y_pred = decisionTree2.predict(X_test)

evaluateAccuracy(y_pred)


# WITH GINI CRITERION AND RANDOM SPLITTER

# create instance of decision tree
decisionTree3 = DecisionTreeClassifier(criterion='gini', splitter="random")
# train the model
decisionTree3.fit(X_train,y_train)

# make predictions using the testing data
y_pred = decisionTree3.predict(X_test)

evaluateAccuracy(y_pred)


# WITH ENTROPY CRITERION AND RANDOM SPLITTER

# create instance of decision tree
decisionTree4 = DecisionTreeClassifier(criterion='entropy', splitter="random")
# train the model
decisionTree4.fit(X_train,y_train)

# make predictions using the testing data
y_pred = decisionTree4.predict(X_test)

evaluateAccuracy(y_pred)


# WITH STANDARD SCALER

# create instance of decision tree
decisionTree5 = DecisionTreeClassifier(criterion='entropy')

# apply standard scaler to the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# train the model
decisionTree5.fit(X_train_std,y_train)

# make predictions using the testing data
y_pred = decisionTree5.predict(X_test_std)

evaluateAccuracy(y_pred)


# WITH EDITED DATASET

#loading dataset without 0's 
phishingEdited = phishing.replace([0], -1) 
# split the dataset into features (X) and targets (y)
X = phishingEdited.drop(["id","having_Sub_Domain","double_slash_redirecting","Result"], axis=1)
y = phishingEdited.Result

# split the dataset into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decisionTree6 = DecisionTreeClassifier(criterion='entropy')

# train the model
decisionTree4.fit(X_train,y_train)

# make predictions using the testing data
y_pred = decisionTree4.predict(X_test)

evaluateAccuracy(y_pred)



