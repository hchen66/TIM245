# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:26:37 2016

@author: Kevin
"""

import pandas as pd
import time
import csv
import numpy as np
import os
import simplejson

data = pd.read_csv('/Users/Kevin/Documents/UCSC/2016Winter/TIM245/Project/train.csv', parse_dates = ['Dates'])
data['hour'] = data['Dates'].dt.hour
data['Year'] = data['Dates'].dt.year
data['Month'] = data['Dates'].dt.month
data['Day'] = data['Dates'].dt.day

data_15 = data[data['Year'].isin([2015])]
data_1505 = data_15[data_15['Month'].isin([5])]
data_11 = data_1505[data_1505['Day'].isin([11])]
data_12 = data_1505[data_1505['Day'].isin([12])]
data_13 = data_1505[data_1505['Day'].isin([13])]

def to_coordiante(df, filename):
    coor = []
    for index in range(len(df)):
        coor.append([df.iloc[index,1], df.iloc[index,2], df.iloc[index,8], df.iloc[index,7]])
    f = open(filename, 'w')
    simplejson.dump(coor, f)
    f.close()
    
to_coordiante(data_11,'data_11.coor')   
to_coordiante(data_12,'data_12.coor')  
to_coordiante(data_13,'data_13.coor')  
