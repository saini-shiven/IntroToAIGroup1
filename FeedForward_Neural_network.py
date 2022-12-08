#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 04:12:48 2022

@author: vijaykesireddy
"""

# library to load dataset (AI.csv)
import pandas as pd
from tensorflow.keras import models, layers # model two
# Preprocessinf steps
from sklearn.model_selection import train_test_split # For training and Testing purposed
from sklearn import preprocessing # Scale our data feature values
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
AI_dataset=pd.read_csv('phishingDataset.csv')
# drop the index column and show the dataset.
AI_dataset=AI_dataset.drop('id',axis=1)


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


# create object of label encoder class
label_encode= preprocessing.LabelEncoder()
# fit and transform training,testing labels
target_of_training_data=label_encode.fit_transform(target_of_training_data)
target_of_testing_data=label_encode.fit_transform(target_of_testing_data)

# start building custom neural network
neural_network=models.Sequential()
# here is the input layer of my neural network
neural_network.add(layers.Dense(128, input_shape=(features_of_training_data.shape[1],), activation='relu'))
# here is the hidden layer of my neural network
neural_network.add(layers.Dense(128, activation='relu'))
# here is the hidden layer of my neural network
neural_network.add(layers.Dense(64, activation='relu'))
# here is the hidden layer of my neural network
neural_network.add(layers.Dense(32, activation='relu'))
# here is the hidden layer of my neural network
neural_network.add(layers.Dense(8, activation='relu'))
# here is the last output layer of my neural network
neural_network.add(layers.Dense(1, activation='sigmoid'))
# check summary of model
print("\nSummary of Neural Network \n")
print(neural_network.summary())

# compile the above neural network with loss function and optimiers
neural_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Start training of model with epoch 20, and both training and validation data
print("\n Start Neural network training \n")
History_of_NN=neural_network.fit(features_of_training_data,target_of_training_data,epochs=20,validation_data=(features_of_testing_data,target_of_testing_data))


# Obtaining predictions of neural network on test dataset
prediction_from_neural_nerwork=neural_network.predict(features_of_testing_data)
# Compile model prediction probabilies
prediction_neural_network=[]
for i in prediction_from_neural_nerwork:
  if i >=0.5: # if probability greater then 0.50
    prediction_neural_network.append(1)
  else: # if probability less then 0.50
    prediction_neural_network.append(0)
print('\n Evalution of Neural network \n')
# # call function.. and show results of Neural Network
Model_Testing(prediction_neural_network,target_of_testing_data)


