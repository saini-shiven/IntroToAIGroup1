#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 04:12:48 2022

@author: vijaykesireddy
"""

# library to load dataset (AI.csv)
import pandas as pd
# Selected AI models Linear Model
from sklearn import linear_model # model one
# Preprocessinf steps
from sklearn.model_selection import train_test_split # For training and Testing purposed
# For the model Evaluations
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# A general function for model testing with accuracy, classification report etc.
def Model_Testing(predicted_labels, actual_labels):
    print("\n")#new line
    print ("Accuracy_test_set: ",accuracy_score(actual_labels,predicted_labels))#accuracy of model
    print("\n")#new line
    print ("Classification_report_of_model : "+"\n", classification_report(actual_labels,predicted_labels))#classificaion report
    print("\n")#new line
    print ("Confusion_Matrix_of model : "+"\n", confusion_matrix(actual_labels,predicted_labels))#confusion metrix



# Start with reading gvien dataset
AI_dataset=pd.read_csv('/Users/vijaykesireddy/Desktop/Website AI/AI.csv')
# drop the index column and show the dataset.
AI_dataset=AI_dataset.drop('index',axis=1)


# Find the information of dataset, here check Datatype of each coulmn
print('Information about data \n \n ')
AI_dataset.info()
# computing basic statistic of our dataset, mean,std,min,max etc.
AI_dataset.describe()

# Next observed the null values, there is no missing value in dataset
AI_dataset[AI_dataset.isnull()].sum()
# print('\n \n')

# First Extract all the features
features_of_Data=AI_dataset.drop('Result',axis=1)
# Second Extract only target column
target_classes=AI_dataset['Result']

# checking the target labels, here we have '1' and '-1' only
print('\n Class distribution. \n',target_classes.value_counts(),' \n')

# Split our dataset features and target labels into training and testing variables 70% use for training 30% testing
features_of_training_data,features_of_testing_data,target_of_training_data,target_of_testing_data=train_test_split(features_of_Data,target_classes,test_size=0.30,random_state=42)

# Creating object of our model
LogisticReg_model = linear_model.LogisticRegression()
# pass the training features and training labels to our model
LogisticReg_model.fit(features_of_training_data,target_of_training_data)
print('The First model training...Done ')

# Get prediction of model using test dataset
prediction_from_logreg=LogisticReg_model.predict(features_of_testing_data)
# call function.. and show results
print('\n Evalution of Logistic Reggression \n')
Model_Testing(prediction_from_logreg,target_of_testing_data)


