# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 13:09:00 2018

@author: sobhan.kumar
"""

import numpy as np
import matplotlib . as mpl
import pandas as pd

dataset = pd.read_csv('Data.csv')
dataset1 = pd.read_csv('Data1.csv')
#from sklearn import preprocessor

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

A =  dataset1.iloc[:, :-1].values
B = dataset1.iloc[:, 3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

imputer =  Imputer(missing_values = 'NaN',strategy = 'most_frequent', axis = 0)
imputer.fit(A[:, :-2])
A[:, :-2] = imputer.transform(A[:, :-2])