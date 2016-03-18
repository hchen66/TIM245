# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:16:51 2016

@author: Kevin
"""

import pandas as pd
import time
import csv
import numpy as np
import os

from sklearn import preprocessing, cross_validation
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sknn.mlp import Classifier, Layer
from sklearn.decomposition import PCA

clear = lambda: os.system('cls')
clear()

train = pd.read_csv('newtrain.csv', parse_dates = ['Dates'])
train = train.head(n=100000)

print train.head()


train['StreetNo'] = train['Address'].apply(lambda x: x.split(' ', 1)[0] if x.split(' ', 1)[0].isdigit() else 0)
#train['Address'] = train['Address'].apply(lambda x: x.split(' ', 1)[1] if x.split(' ', 1)[0].isdigit() else x)
train['hour'] = train['Dates'].dt.hour
train['evening'] = train['Dates'].dt.hour.isin([18,19,20,21,22,23,0,1,2,3,4,5,6])
train['Year'] = train['Dates'].dt.year
#train = train[train['Year'].isin([2011,2012,2013,2014,2015])]
train['Month'] = train['Dates'].dt.month




start = time.time()
lenth = len(train)
cur = 0

print '  -> processing time:', time.time() - start
#print train.head()
print len(set(train['StreetNo'])), len(set(train['Address']))

le = LabelEncoder()
crime = le.fit_transform(train.Category)

hour = pd.get_dummies(train.hour)
district = pd.get_dummies(train.PdDistrict)
StreetNo = pd.get_dummies(train.StreetNo)
evening = pd.get_dummies(train.evening)
ContainOf = pd.get_dummies(train.AddressContainOf)
Year = pd.get_dummies(train.Year)
Month = pd.get_dummies(train.Month)

train_data = pd.concat([hour, district, StreetNo, evening, ContainOf, train['X'], train['Y']], axis=1)
train_data['crime'] = crime
crime_data = train_data.iloc[:,:-1]
crime_label = train_data['crime']

classifiers = [
    BernoulliNB(),
    RandomForestClassifier(max_depth=10, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=12, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=14, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=16, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=18, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=20, n_estimators=1024, n_jobs=-1),
    RandomForestClassifier(max_depth=22, n_estimators=1024, n_jobs=-1),
    KNeighborsClassifier(n_neighbors=100, weights='distance', algorithm='ball_tree', leaf_size=100, p=10, metric='minkowski'),
    #XGBClassifier(max_depth=16,n_estimators=1024),
    GradientBoostingClassifier(n_estimators=10, learning_rate=1.0,max_depth=5, random_state=0),
    AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=8), algorithm="SAMME.R", n_estimators=128),
    ]
    
#print train.head()
    
newClassifiers = [
    BernoulliNB(),
    RandomForestClassifier(max_depth=16, n_estimators=1024, n_jobs=-1),
    GradientBoostingClassifier(max_depth=16, n_estimators=1024)
    #KNeighborsClassifier(n_neighbors=50, weights='distance', algorithm='ball_tree', leaf_size=100, p=10, metric='minkowski', n_jobs=-1),
    ]

 
#[train_d, test_d, train_labels, test_labels] = cross_validation.train_test_split(crime_data, crime_label, test_size=0.2, random_state=20160217)
skf = cross_validation.StratifiedKFold(crime_label, n_folds=2, random_state=20160217, shuffle=True)
for train_index, test_index in skf:
    train_d, test_d = crime_data.iloc[train_index,:], crime_data.iloc[test_index,:]
    train_labels, test_labels = crime_label[train_index], crime_label[test_index]
    print train_d.shape, test_d.shape
    for classifier in classifiers:
        print classifier.__class__.__name__
        start = time.time()
        classifier.fit(train_d, train_labels)
        print '  -> Training time:', time.time() - start
                        
        start = time.time()        
        #score_result = classifier.score(test_d, test_labels)
        #print '  -> caluclate score time', time.time() - start
                        
        start = time.time()
        predicted = np.array(classifier.predict_proba(test_d))
        print '  -> predict_proba time:', time.time() - start
                        
        start = time.time()
        log_result = log_loss(test_labels, predicted)
        print '  -> calculate log_loss time:', time.time() - start        
                        
        #print "score = ", score_result, "log loss = ",log_result
        print "log_loss = ", log_result
    