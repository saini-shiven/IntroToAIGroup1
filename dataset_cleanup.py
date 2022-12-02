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

#DEALING WITH DUPLICATE ENTRIES
newdf = df.drop_duplicates()
print(df)