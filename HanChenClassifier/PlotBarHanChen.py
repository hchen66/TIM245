# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:48:20 2016

@author: Kevin
"""
import pandas as pd
import time
import csv
import numpy as np
import os
import operator
import string
import seaborn as sns

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

import matplotlib.pyplot as plt


def plothelper(df, title):
    p = ('Set2', 'Paired', 'colorblind', 'husl','Set1', 'coolwarm', 'RdYlGn', 'spectral')
    color = sns.color_palette(np.random.choice(p), len(df))
    bar   = df.plot(kind='barh',title=title,fontsize=8,figsize=(10,8),stacked=False,width=1,color=color,)
    plt.show()
    
def plot(df, column, title, items=0):
    lower_case     = operator.methodcaller('lower')
    df.columns     = df.columns.map(lower_case)
    by_col         = df.groupby(column)
    col_freq       = by_col.size()
    col_freq.index = col_freq.index.map(string.capwords)

    col_freq.sort(ascending=True, inplace=True)

    plothelper(col_freq[slice(-1, - items, -1)], title)

data = pd.read_csv('/Users/Kevin/Documents/UCSC/2016Winter/TIM245/Project/train.csv')

plot(data, 'category',   'Top Crime Categories')
#plot(data, 'resolution', 'Top Crime Resolutions')
#plot(data, 'pddistrict', 'PdDescript')
#plot(data, 'dayofweek',  'Days of the Week')
#plot(data, 'address',    'Top Crime Locations', items=15)
#plot(data, 'descript',   'Descriptions', items=15)
