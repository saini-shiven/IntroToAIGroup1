# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd
import seaborn as sns
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

#Percentage of class values
plt.figure(figsize=(13, 6))
ax = sns.barplot(x=phishing['Result'], y=phishing['Result'], data=df, estimator=lambda x: len(x) / len(df) * 100, color = 'Red')
ax.set(ylabel="Percent");
ax.set_title('The Percentage of Phishing vs Legitimate Webites', size = 15);
plt.savefig('resultpercentageplot.png', dpi=300, bbox_inches='tight');

#Correlation between variables
plt.figure(figsize=(35, 25))
corr = df.corr()
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(corr, mask = mask, vmin=-1, vmax=1, annot=True, cmap = 'viridis')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=12);
plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')