# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# load phishing dataset into dataframe
phishing = pd.read_csv("phishingDataset.csv")
df = phishing.drop(["id","Result"], axis=1)

# split the dataset into features (X) and targets (y)
#X = phishing.drop(["id","Result"], axis=1)
#y = phishing.Result

phishing.info()

# defining phishing websites
P = phishing[phishing.Result == -1]

# non-phishing websites
NP = phishing[phishing.Result == 1]

pd.crosstab(phishing['Result'],
            phishing['age_of_domain'],
            colnames=["Age of Domain"],
            rownames=["Result (-1=Phishing,1=Legitimate)"]
            ).plot.bar(stacked=False)