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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sknn.mlp import Classifier, Layer
from sklearn.decomposition import PCA

clear = lambda: os.system('cls')
clear()

train = pd.read_csv('../train.csv', parse_dates = ['Dates'])

#train = train.head(n=10)
train['AddressContainOf'] = 0


start = time.time()
lenth = len(train)
cur = 0
for index in range(len(train)):
    if(index > 5000 and index/5000.0 > cur):
        print "pre-processing the", cur*5000, "th data"
        cur += 1
    if(train.iloc[index,6].find('/') == -1):
        train.iloc[index,9] = 1

pca = PCA(n_components=2)
coor = []
for index in range(len(train)):
    coor.append([train['X'].iloc[index], train['Y'].iloc[index]])
pca.fit(coor)
pca_transform = pca.transform(coor)
cur = 0
for index in range(len(train)):
    if(index > 5000 and index/5000.0 > cur):
        print "pre-processing the", cur*5000, "th data"
        cur += 1
    train.iloc[index,7] = pca_transform[index][0]
    train.iloc[index,8] = pca_transform[index][1]
    
train.to_csv('newtrain.csv')
    
    
    