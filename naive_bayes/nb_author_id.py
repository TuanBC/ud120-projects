#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

# Step 1: Extract tf-idf feature vectors from the dataset
features_train, features_test, labels_train, labels_test = preprocess()
# no. of Chris training emails: 7936
# no. of Sara training emails: 7884

# Shape of feature vector
print(features_train.shape)
#(15820, 3785)
print(len(labels_train))
# 15820
print(features_test.shape)
# (1758, 3785)
print(len(labels_test))
# 1758

# Step 2: Train a Gaussian Naive Bayes Classifier
start=time()
clf = GaussianNB()
clf.fit(features_train, labels_train)
end=time()
print ('Training time: ', end-start)
# Training time:  1.038116455078125

#Step 3: Predict labels from the test set
start=time()
pdt_label = clf.predict(features_test)
end=time()
print ('Testing time: ', end-start)
# Testing time:  0.1615900993347168

#Step 4; Get the accuracy by comparing predicted labels with the ones from the dataset
print('Accuracy:', accuracy_score(labels_test, pdt_label))
# Accuracy: 0.973265073948

