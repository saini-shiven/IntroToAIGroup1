# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:21:19 2022

@author: saini
"""

import os
import platform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sb
from scipy import stats



#load data
df = pd.read_csv("phishingDataset.csv")


#heatmap of data, plotting where/if we're missing data
plt.figure(figsize = (8,6))
sb.heatmap(df.isnull(), cbar=False , cmap = 'magma')


#Show sum of missing values within each column
print(df.isnull().sum())
#Selecting our limits
q_low = df.drop('Result').quantile(0.01)
q_hi  = df.drop('Result', axis=1).quantile(0.99)
#Filterng outliers
df_filtered = df[(df["Result"] < q_hi) & (df["Result"] > q_low)]
#DOES NOTHING LMAO
print(df_filtered)

#DEALING WITH DUPLICATE ENTRIES
