#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 18:56:14 2019

@author: jaimeiglesias

Working Directory: /Users/jaimeiglesias/Documents/Learning/0.Data Science/0.Projects/2.App UX

Purpose: To analyze the app behavior Dataset in order to publish an 
         intelligence report on Github.

Index: 
    * Data preprocessing: flagging and imputing nas, visual EDA, 
    flagging outliers, and transform categorical variables into dummies.
"""

# Importing packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import shutil


# Loading dataset
app_df = pd.read_excel('0.Mobile_App_Survey_Data.xlsx')

app_df.head()
app_df.shape
app_df.info()
app_df_describe = app_df.describe()


# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)



###############################################################################
## 1. Data Preprocessing
###############################################################################

#######################################
# 1.1. Flagging, filling NAs, data cleaning
#######################################

# Checking for NAs
app_df.isnull().any().sum()


# q2r10 and q11 inconsistency checks
(app_df['q11'] == 6).sum()
(app_df['q2r10'] == 1).sum()


# q13 order flip
q13_replace = {1:4,
               2:3,
               3:2,
               4:1}

app_df.iloc[:, 25:37].replace(q13_replace, inplace = True)


# q20s order flip
q20_replace = {1:6,
               2:5,
               3:4,
               4:3,
               5:2,
               6:1}

app_df.iloc[:, 37:77].replace(q20_replace, inplace = True)


#######################################
# 1.2. Visual EDA
#######################################

# Viewing the first few rows of the data
app_df.head(n = 5)


"""
    * 8 variables of type int64
    * no missing values
    * no negative values
    * many distributions seem skewed
"""


########################
# Histograms
########################

# List with numerical variables
num_cols = [col for col in app_df][2:77]


# List with categorical variables
cat_cols = [col for col in app_df][-11:]
cat_cols.insert(0,'q1')

app_df['q1'].value_counts()
app_df['q48'].value_counts()
app_df['q49'].value_counts()
app_df['q54'].value_counts()
app_df['q56'].value_counts()


# Creating folder in current directory to store graphs
cwd = os.getcwd()

if os.path.exists(cwd + '/1.2.EDA Graphs'):
  shutil.rmtree(cwd + '/1.2.EDA Graphs')
  
os.makedirs(cwd + '/1.2.EDA Graphs')  
graph_path = cwd+'/1.2.EDA Graphs/'


# 1.2.1. Histograms numerical variables
f, axes = plt.subplots(11, 7, figsize = (40, 60))
for i, e in enumerate(num_cols):
    sns.distplot(app_df[e],
                 bins = 'fd',
                 kde = True,
                 rug = False,
                 ax = axes[i // 7][i % 7])
    plt.xlabel(e)
    
plt.savefig(graph_path + '1.Histograms numerical.png')


# 1.2.2. Histograms categorical variables
f, axes = plt.subplots(3, 4, figsize = (40, 60))
for i, e in enumerate(cat_cols):
    sns.distplot(app_df[e],
                 bins = 'fd',
                 kde = True,
                 rug = False,
                 ax = axes[i // 4][i % 4])
    plt.xlabel(e)
    
plt.savefig(graph_path + '2.Histograms categorical.png')



#######################################
# 1.2. Correlation heatmap of scaled data
#######################################


# Scaling the wholesale app dataset
app_features = app_df.iloc[ : , : ]



# Scaling using StandardScaler()
scaler = StandardScaler()
scaler.fit(app_features)
X_scaled = scaler.transform(app_features)

X_scaled_df = pd.DataFrame(X_scaled)


# Creating heatmap
fig, ax = plt.subplots(figsize = (40, 40))
df_corr = app_df.corr().round(2)
sns.heatmap(df_corr,
            cmap = 'coolwarm',
            square = True,
            annot = True)


plt.savefig(graph_path + '3.App correlations.png')
plt.show()



app_df.to_excel('1.3.App processed.xlsx')