# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 14:08:06 2022

@author: saini
"""

#libraries for the task
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

#load data
df = pd.read_csv("phishingDataset.csv", na_values=['NaN'])

#assigning x and y into features and labels respectfully; Labels being what we want to predict, and features being what we use to predict
X = df.drop('Result', axis=1)
y = df.Result

# split the data 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=60)

#set up a model
ppn = Perceptron(max_iter=40,tol=0.001,eta0=1)

#Train the model
ppn.fit(X_train,y_train)

# Make predication
y_pred = ppn.predict(X_test)

# Evaluate accuracy
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Confusion matrix
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cm,[-1,1])
