# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 04:52:42 2025

@author: syama
"""

import pandas as pd
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("bZIP_HI_IUPred.txt",sep='\t',header=0)
df=df.loc[0:19539,:]
df.head()
mask = df['DisorderTendency'].isna() | np.isinf(df['DisorderTendency'])

if mask.any():
    print("There is at least one NaN or Inf in the column.")
    true_indices = mask[mask].index
    print(true_indices)
else:
    print("No NaN or Inf values found in the column.")

#replace na withmean
df['DisorderTendency'] = df['DisorderTendency'].fillna(df['DisorderTendency'].mean())

######################################################################
# Full dataset correlation
pearson_corr, _ = pearsonr(df['Hydrophobicity'], df['DisorderTendency'])
spearman_corr, _ = spearmanr(df['Hydrophobicity'], df['DisorderTendency'])
print("Pearson Correlation:", pearson_corr)
print("Spearman Correlation:", spearman_corr)

binding_df = df[df['binding'] == 1]
pearson_binding, _ = pearsonr(binding_df['Hydrophobicity'], binding_df['DisorderTendency'])
pearson_binding

#Plot the correlation
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x='Hydrophobicity', y='DisorderTendency', data=df)
plt.title("Hydrophobicity vs Disorder Tendency")
plt.show()


#####################



# Separate by binding region
binding_df= df[df['binding'] == 1]
non_binding_df = df[df['binding'] == 0]
data_binding = binding_df[['Hydrophobicity','DisorderTendency']]
data_binding.shape
corr_binding = binding_df['Hydrophobicity'].corr(binding_df['DisorderTendency'])
corr_non_binding = non_binding_df['Hydrophobicity'].corr(non_binding_df['DisorderTendency'])
Full_correlation = data_binding.corr(method='pearson')
Full_correlation
print(f"Correlation (binding region): {corr_binding:.3f}")
print(f"Correlation (non-binding region): {corr_non_binding:.3f}")


sns.lmplot(x="Hydrophobicity", y="DisorderTendency", hue="binding", lowess=True, data=df, height=10, aspect=1.8);

#non linear plot############
plt.figure(figsize=(12, 10))

for label, group_df in df.groupby("binding"):
    sns.regplot(
        x="Hydrophobicity", 
        y="DisorderTendency", 
        data=group_df, 
        lowess=True, 
        label=label
    )

plt.legend(title="Binding Region")
plt.title("Lowess Smoothing by Binding Region")
plt.show()
#####################



sns.set(style="whitegrid")

# Plotting with a smooth line (regression line)
plt.figure(figsize=(10, 6))

sns.lineplot(
    x='Hydrophobicity', y='DisorderTendency',
    data=binding_df, label='Binding Region', color='blue', estimator='mean'
)
sns.lineplot(
    x='Hydrophobicity', y='DisorderTendency',
    data=non_binding_df, label='Non-Binding Region', color='orange', estimator='mean'
)

plt.title("Hydrophobicity vs Disorder Tendency")
plt.xlabel("Hydrophobicity")
plt.ylabel("Disorder Tendency")
plt.legend()
plt.tight_layout()
plt.show()
####################
binding_corr, _ = spearmanr(binding_df['Hydrophobicity'], binding_df['DisorderTendency'])
nonbinding_corr, _ = spearmanr(non_binding_df['Hydrophobicity'], non_binding_df['DisorderTendency'])

print(f"Spearman Correlation in Binding Region: {binding_corr:.3f}")
print(f"Spearman Correlation in Non-Binding Region: {nonbinding_corr:.3f}")


##############
#4. Mutual Information Score
#This can detect any kind of dependency, not just linear or monotonic.


from sklearn.feature_selection import mutual_info_regression

mi = mutual_info_regression(df[['Hydrophobicity']], df['DisorderTendency'])
print(f"Mutual Information: {mi[0]:.3f}")
