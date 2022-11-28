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

#Evaluate accuracy
print('Accuracy without any folds: %.2f' % accuracy_score(y_test, y_pred))

#or use k-fold cross-validation
kf = KFold(5, shuffle=True)
# Mention that 10 didn't help accuracy

#with standardisation
print("")
print("With standardisation and 5 folds:")

sc = StandardScaler()

fold = 1
# The data is split five ways, for each fold, the 
# Perceptron is trained, tested and evaluated for accuracy
for train_index, validate_index in kf.split(X,y):
    sc.fit(X.iloc[train_index])
    X_train_std = sc.transform(X.iloc[train_index])
    X_test_std = sc.transform(X.iloc[validate_index])
    ppn.fit(X_train_std,y.iloc[train_index])
    y_test = y.iloc[validate_index]
    y_pred = ppn.predict(X_test_std)
    print(f"Fold #{fold}, Training Size: {len(X.iloc[train_index])}, Validation Size: {len(X.iloc[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 1

    
#Confusion matrix
def plot_confusion_matrix(cm, names, title='CAVEMAN AI', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label (-1; Phishing, 1; Non-Phishing)')
    plt.xlabel('Predicted label (-1; Phishing, 1; Non-Phishing)')
    
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plot_confusion_matrix(cm,[-1,1])
