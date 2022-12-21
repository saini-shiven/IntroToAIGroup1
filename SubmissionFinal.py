# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:51:04 2022

@author: AI group 1
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
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import ConfusionMatrixDisplay 
from sklearn import linear_model
  

#load overall data
df = pd.read_csv("phishingDataset.csv", na_values=['NaN'])
#loading dataset without 0's
dfEdited = df.replace([0], -1)

print("AI Group Project, group 1; see plots for graphical representations of dataset models")
print("")

"""
Perceptron Model
@author: Shiven Saini
"""
#assigning x and y into features and labels respectfully; Labels being what we want to predict, and features being what we use to predict
X = df.drop('Result', axis=1)
y = df.Result

# split the data 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=60)

#set up a model
ppn = Perceptron(max_iter=40,tol=0.001,eta0=1)

#Train the model
ppn.fit(X_train,y_train)

# Make predication
y_pred = ppn.predict(X_test)

#Evaluate accuracy
print('Accuracy of Perceptron without any folds: %.2f' % accuracy_score(y_test, y_pred))

#or use k-fold cross-validation
kf = KFold(5, shuffle=True)

#with standardisation
print("")
print("With standardisation and 5 folds:")

sc = StandardScaler()
totAcc = 0

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
    fold += 1
    totAcc += accuracy_score(y_test, y_pred)

print('Mean accuracy of perceptron with original dataset: %.2f' % (totAcc/5))

#Plotting confusion matrix    
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Perceptron Confusion Matrix for Original Dataset')
plt.colorbar(fraction=0.05)
tick_marks = np.arange(len([-1, 1]))
plt.xticks(tick_marks, [-1,1], rotation=45)
plt.yticks(tick_marks, [-1,1])
plt.tight_layout()
plt.ylabel('True label (-1; Phishing, 1; Non-Phishing)')
plt.xlabel('Predicted label (-1; Phishing, 1; Non-Phishing)')

#Repeating this Process with our second dataset
#assigning x and y into features and labels respectfully; Labels being what we want to predict, and features being what we use to predict
X = dfEdited.drop('Result', axis=1)
y = dfEdited.Result

#Using k-fold cross validation
kf = KFold(5, shuffle=True)
# Mention that 10 didn't help accuracy

sc = StandardScaler()
totAcc = 0

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
    fold += 1
    totAcc += accuracy_score(y_test, y_pred)

print('Mean accuracy of perceptron with edited dataset: %.2f' % (totAcc/5))

#Plotting confusion matrix    
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Perceptron Confusion Matrix for edited Dataset')
plt.colorbar(fraction=0.05)
tick_marks = np.arange(len([-1, 1]))
plt.xticks(tick_marks, [-1,1], rotation=45)
plt.yticks(tick_marks, [-1,1])
plt.tight_layout()
plt.ylabel('True label (-1; Phishing, 1; Non-Phishing)')
plt.xlabel('Predicted label (-1; Phishing, 1; Non-Phishing)')


    
"""
SVM Model
@author: Shiven Saini
"""
#method to plot confusion matrices
def plot_confusion_matrix(cm, names, title='SVM Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
#plotting the data as a svc decision function
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
                levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                    model.support_vectors_[:, 1],
                    s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def plot_svm(N=10, ax=None):
    X, y = make_blobs(n_samples=200, centers=2,
                      random_state=0, cluster_std=0.60)
    X = X[:N]
    y = y[:N]
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)
    
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1, 6)
    plot_svc_decision_function(model, ax)


#USING FIRST 400 ROWS WORKS BETTER THAN RANDOM SAMPLE
#See repo for proof of pre-analysis
SVMdf = df.head(400)

#shuffle the data
SVMdf = SVMdf.reindex(np.random.permutation(SVMdf.index))

#list of columns that aren't result
cols = []
for x in SVMdf.columns:
    if x != 'Result':
        cols.append(x)
        
#defining x and y 
X = SVMdf[cols].values
y = SVMdf['Result'].values


#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 


#Plotting the first 60 and 120 data points with line of best fit
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))

#Using an RBF kernel rather than a linear one;
#RBF - Radial Basis function
#Embraces approximations to allow for better scaling
#to large datasets
svm_model = SVC(kernel='rbf', C=100).fit(X, y)

#Predicting our results based off of the test data
y_pred = svm_model.predict(X_test)

print("")
print('Accuracy of SVM when trained against original dataset: %.2f' % accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

#Non-normalised confusion matrix
plt.figure()
plot_confusion_matrix(cm, [-1,1], title='SVM Non-Normalized Confusion matrix for original Dataset')
plt.show()

#normalised confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, [-1,1], title='SVM Normalized Confusion matrix for original Dataset')
plt.show()

#Repeating this process with edited dataset 

#USING FIRST 400 ROWS WORKS BETTER THAN RANDOM SAMPLE
#See repo for proof of pre-analysis
SVMEditeddf = dfEdited.head(400)

#shuffle the data
SVMEditeddf = SVMEditeddf.reindex(np.random.permutation(SVMEditeddf.index))

#list of columns that aren't result
cols = []
for x in SVMEditeddf.columns:
    if x != 'Result':
        cols.append(x)
        
#defining x and y 
X = SVMEditeddf[cols].values
y = SVMEditeddf['Result'].values


#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 


#Plotting the first 60 and 120 data points with line of best fit
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for axi, N in zip(ax, [60, 120]):
    plot_svm(N, axi)
    axi.set_title('N = {0}'.format(N))

#Using an RBF kernel rather than a linear one;
#RBF - Radial Basis function
#Embraces approximations to allow for better scaling
#to large datasets
svm_model = SVC(kernel='rbf', C=100).fit(X, y)

#Predicting our results based off of the test data
y_pred = svm_model.predict(X_test)

print('Accuracy of SVM when trained against edited dataset (0s replaced with -1s): %.2f' % accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

#Non-normalised confusion matrix
plt.figure()
plot_confusion_matrix(cm, [-1,1], title='SVM Non-Normalized Confusion matrix for edited Dataset')
plt.show()

#normalised confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized, [-1,1], title='SVM Normalized Confusion matrix for edited Dataset')
plt.show()

"""
KNN Model
@author: Zainab Mayet
"""


# load phishing dataset into dataframe 
phishing = pd.read_csv("phishingDataset.csv") 
#loading dataset without 0's 
phishingEdited = phishing.replace([0], -1) 

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
knn_model = KNeighborsClassifier(n_neighbors=15) 

#fit and test kNN 
knn_model.fit(X_train, y_train) 
y_pred_knn = knn_model.predict(X_test) 
print('Mean Accuracy of KNN with original dataset: %.3f' % accuracy_score(y_test, y_pred_knn)) 

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

#repeat for dataset without 0s 

# Select all columns but price (which is the target) as data features 
X = phishingEdited[ phishingEdited.columns[phishingEdited.columns!='Result'] ]  
# Select price as target 

y = phishingEdited['Result'] 

#split into testing and training 

X_train, X_test, y_train, y_test = train_test_split(     
    X, y, test_size=0.25, random_state=42)  

  

#For the following models, some experimentation has been performed 
#in order to find the appropriate parameter for the models. 
#This isn't given here, but you might try varying the values. 
#You might well want to plot the curves that your find. 

  

#set up kNN 
knn_model = KNeighborsClassifier(n_neighbors=15) 

#fit and test kNN 
knn_model.fit(X_train, y_train) 
y_pred_knn = knn_model.predict(X_test) 
print('Mean Accuracy of KNN with dataset without 0s: %.3f' % accuracy_score(y_test, y_pred_knn)) 


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

#repeat for dataset without having_Sub_Domain & double_slash_redirecting

# Select all columns but price (which is the target) as data features 
X = phishingEdited[ phishingEdited.columns[phishingEdited.columns!='Result']  & phishing.columns[phishing.columns!='having_Sub_Domain'] & phishing.columns[phishing.columns!='double_slash_redirecting']]  
# Select price as target 

y = phishingEdited['Result'] 

#split into testing and training 

X_train, X_test, y_train, y_test = train_test_split(     
    X, y, test_size=0.25, random_state=42)  

  

#For the following models, some experimentation has been performed 
#in order to find the appropriate parameter for the models. 
#This isn't given here, but you might try varying the values. 
#You might well want to plot the curves that your find. 

  

#set up kNN 
knn_model = KNeighborsClassifier(n_neighbors=15) 

#fit and test kNN 
knn_model.fit(X_train, y_train) 
y_pred_knn = knn_model.predict(X_test) 
print('Mean Accuracy of KNN with dataset without two features: %.3f' % accuracy_score(y_test, y_pred_knn)) 


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

"""
Random Forest Model
@author: Zainab Mayet
"""

# load phishing dataset into dataframe
phishing = pd.read_csv("phishingDataset.csv")
#loading dataset without 0's 
phishingEdited = phishing.replace([0], -1) 

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


#set up randon forest
rf_model = RandomForestClassifier(n_estimators = 15)



#fit and test kNN
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('Mean Accuracy of RF with original dataset: %.3f' % accuracy_score(y_test, y_pred_rf))


# train the model
rf_model.fit(X_train,y_train)

# make predictions using the testing data
y_pred = rf_model.predict(X_test)




def evaluateAccuracy(predictions):
    
    # produce confusion matrix
    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
    display.plot()
    plt.show()

evaluateAccuracy(y_pred)


#Repeat with datset without 0s

# Select all columns but price (which is the target) as data features
X = phishingEdited[ phishingEdited.columns[phishingEdited.columns!='Result'] ] 
# Select price as target
y = phishingEdited['Result']

#split into testing and training
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 

#For the following models, some experimentation has been performed
#in order to find the appropriate parameter for the models.
#This isn't given here, but you might try varying the values.
#You might well want to plot the curves that your find.


#set up randon forest
rf_model = RandomForestClassifier(n_estimators = 15)



#fit and test kNN
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('Mean Accuracy of RF with dataset without 0s: %.3f' % accuracy_score(y_test, y_pred_rf))


# train the model
rf_model.fit(X_train,y_train)

# make predictions using the testing data
y_pred = rf_model.predict(X_test)


def evaluateAccuracy(predictions):
    
    # produce confusion matrix
    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
    display.plot()
    plt.show()

evaluateAccuracy(y_pred)


#repeat for dataset without having_Sub_Domain & double_slash_redirecting


# Select all columns but price (which is the target) as data features
X = phishingEdited[ phishingEdited.columns[phishingEdited.columns!='Result'] & phishing.columns[phishing.columns!='having_Sub_Domain'] & phishing.columns[phishing.columns!='double_slash_redirecting'] ] 
# Select price as target
y = phishingEdited['Result']

#split into testing and training
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 

#For the following models, some experimentation has been performed
#in order to find the appropriate parameter for the models.
#This isn't given here, but you might try varying the values.
#You might well want to plot the curves that your find.


#set up randon forest
rf_model = RandomForestClassifier(n_estimators = 15)



#fit and test kNN
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('Mean Accuracy of RF with dataset without two features: %.3f' % accuracy_score(y_test, y_pred_rf))


# train the model
rf_model.fit(X_train,y_train)

# make predictions using the testing data
y_pred = rf_model.predict(X_test)


def evaluateAccuracy(predictions):
    
    # produce confusion matrix
    cm = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
    display.plot()
    plt.show()

evaluateAccuracy(y_pred)

"""
Decision Tree
@author: Andreas Salcedo
"""
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
    print('Accuracy of decision tree with original dataset: %.2f' % accuracy)
    
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

accuracy = accuracy_score(y_test, y_pred)
# rounded to 2 significant figures
print('Accuracy of decision tree with edited dataset: %.2f' % accuracy)

# produce confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
display.plot()
plt.show()

"""
Naive Bayes
@author: Andreas Salcedo
"""
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
print('Accuracy of Naive Bayes (Gaussian) with original dataset: %.2f' % accuracy)

# produce confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
display.plot()
plt.show()

# BERNOULLI CLASSIFIER

# split the dataset into a training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create gaussian naive bayes model
model = BernoulliNB()
model.fit(X, y);

# make predictions using the testing data
y_pred = model.predict(X_test)

# calculate accuracy of model
accuracy = accuracy_score(y_test, y_pred)
# rounded to 2 significant figures
print('Accuracy of Naive Bayes (Bernoulli) with original dataset: %.2f' % accuracy)

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
print('Accuracy of Naive Bayes with original dataset (Standard Scaler): %.2f' % accuracy)

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
print('Accuracy of Naive Bayes with edited dataset: %.2f' % accuracy)

# produce confusion matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(["Phishing","Non-Phishing"]))
display.plot()
plt.show()

"""
Logistic Regression

@author: vijaykesireddy - adapted for submission by Shiven Saini
"""

# A general function for model testing with accuracy, classification report etc.
def Model_Testing(predicted_labels, actual_labels):
    print("\n")#new line
    print ("Accuracy: ",accuracy_score(actual_labels,predicted_labels))#accuracy of model
    

# drop the index column and show the dataset.
df=df.drop('id',axis=1)

# First Extract all the features
features_of_Data=df.drop('Result',axis=1)

# Second Extract only target column
target_classes=df['Result']

# Split our dataset features and target labels into training and testing variables 70% use for training 30% testing
features_of_training_data,features_of_testing_data,target_of_training_data,target_of_testing_data=train_test_split(features_of_Data,target_classes,test_size=0.30,random_state=42)

# Creating object of our model
LogisticReg_model = linear_model.LogisticRegression()
# pass the training features and training labels to our model
LogisticReg_model.fit(features_of_training_data,target_of_training_data)

# Get prediction of model using test dataset
prediction_from_logreg=LogisticReg_model.predict(features_of_testing_data)
# call function.. and show results
print('\n Evalution of Logistic Reggression with original dataset')
Model_Testing(prediction_from_logreg,target_of_testing_data)


#Repeating the process with edited dataset

# drop the index column and show the dataset.
df=dfEdited.drop('id',axis=1)

# First Extract all the features
features_of_Data=df.drop('Result',axis=1)

# Second Extract only target column
target_classes=df['Result']

# Split our dataset features and target labels into training and testing variables 70% use for training 30% testing
features_of_training_data,features_of_testing_data,target_of_training_data,target_of_testing_data=train_test_split(features_of_Data,target_classes,test_size=0.30,random_state=42)

# Creating object of our model
LogisticReg_model = linear_model.LogisticRegression()
# pass the training features and training labels to our model
LogisticReg_model.fit(features_of_training_data,target_of_training_data)

# Get prediction of model using test dataset
prediction_from_logreg=LogisticReg_model.predict(features_of_testing_data)
# call function.. and show results
print('\n Evalution of Logistic Reggression with edited dataset')
Model_Testing(prediction_from_logreg,target_of_testing_data)
