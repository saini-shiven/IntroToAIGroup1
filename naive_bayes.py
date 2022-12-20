#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:09:15 2022

@author: andreas
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import pandas as pd

# load phishing dataset into dataframe
phishing = pd.read_csv("phishingDataset.csv")

# split the dataset into features (X) and targets (y)
X = phishing.drop(["id","Result"], axis=1)
y = phishing.Result

# split the dataset into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create gaussian naive bayes model
model = GaussianNB()
model.fit(X, y);

# make predictions using the testing data
y_pred = model.predict(X_test)

# calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
# rounded to 2 significant figures
print('Accuracy: %.3f' % accuracy)

# produce confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
display.plot()
plt.show()


# WITH STANDARD SCALER

# create instance of decision tree
model2 = GaussianNB()

# apply standard scaler to the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)

# train the model
model2.fit(X_train_std,y_train)

# make predictions using the testing data
y_pred = model2.predict(X_test_std)

# calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
# rounded to 2 significant figures
print('Accuracy: %.3f' % accuracy)

# produce confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
display.plot()
plt.show()


# WITH EDITED DATASET

#loading dataset without 0's 
phishingEdited = phishing.replace([0], -1) 
# split the dataset into features (X) and targets (y)
X = phishingEdited.drop(["id","having_Sub_Domain","double_slash_redirecting","Result"], axis=1)
y = phishingEdited.Result

# split the dataset into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model3 = GaussianNB()

# train the model
model3.fit(X_train,y_train)

# make predictions using the testing data
y_pred = model3.predict(X_test)

# calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
# rounded to 2 significant figures
print('Accuracy: %.3f' % accuracy)

# produce confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
display.plot()
plt.show()


