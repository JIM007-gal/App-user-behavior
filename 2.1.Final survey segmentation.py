#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:30:51 2019

@author: jaimeiglesias

Working Directory: /Users/jaimeiglesias/Documents/Learning/0.Data Science/0.Projects/2.App UX

Purpose: To analyze the App UX Dataset in order to publish an 
         intelligence report on Github.
         
Index: * PCA
       * K Means Clustering
       * Visual representations
"""

# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import os
import shutil


# Load preprocessed dataset
app_df = pd.read_excel('1.3.App processed.xlsx')


# Redefine cwd
cwd = os.getcwd()


    
###############################################################################
# 1. PCA with normalized variables
###############################################################################

# Creating folder in current directory to store graphs
cwd = os.getcwd()

if os.path.exists(cwd + '/2.1.PCA'):
  shutil.rmtree(cwd + '/2.1.PCA')
  
os.makedirs(cwd + '/2.1.PCA')  
graph_path = cwd+'/2.1.PCA/'


## Remove demographic information
app_features_reduced = app_df.iloc[ : , 2:77]


## Scale to get equal variance
scaler = StandardScaler()
scaler.fit(app_features_reduced)
X_scaled_reduced = scaler.transform(app_features_reduced)
X_scaled_reduced
pd.np.var(X_scaled_reduced)
## Run PCA without limiting the number of components
app_pca_reduced = PCA(n_components = None,
                      random_state = 508)


app_pca_reduced.fit(X_scaled_reduced)
X_pca_reduced = app_pca_reduced.transform(X_scaled_reduced)


## Analyze the scree plot to determine how many components to retain
fig, ax = plt.subplots(figsize=(10, 8))

features = range(app_pca_reduced.n_components_)


plt.plot(features,
         app_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')


plt.title('Reduced Wholesale app Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features)

plt.savefig(graph_path + '1.Reduced Normalized Scree Plot.png')
plt.show()



print(f"""

Normally, we may want to set a threshold at explaining 80% of the variance
in the dataset.

Right now we have the following:
    1 Principal Component : {app_pca_reduced.explained_variance_ratio_[0].round(2)}
    2 Principal Components: {(app_pca_reduced.explained_variance_ratio_[0] + app_pca_reduced.explained_variance_ratio_[1]).round(2)}
    3 Principal Components: {(app_pca_reduced.explained_variance_ratio_[0] + app_pca_reduced.explained_variance_ratio_[1] + app_pca_reduced.explained_variance_ratio_[2]).round(2)}
    4 Principal Components: {(app_pca_reduced.explained_variance_ratio_[0] + app_pca_reduced.explained_variance_ratio_[1] + app_pca_reduced.explained_variance_ratio_[2] + app_pca_reduced.explained_variance_ratio_[3]).round(2)}
    5 Principal Components: {(app_pca_reduced.explained_variance_ratio_[0] + app_pca_reduced.explained_variance_ratio_[1] + app_pca_reduced.explained_variance_ratio_[2] + app_pca_reduced.explained_variance_ratio_[3] + app_pca_reduced.explained_variance_ratio_[4]).round(2)}    

""")



## Run PCA again based on the desired number of components
app_pca_reduced = PCA(n_components = 5,
                      random_state = 508)

app_pca_reduced.fit(X_scaled_reduced)


## Analyze factor loadings to understand principal components
"""
    Since the variance amongst factors is getting "bundled" into principal
    components, it is a good idea to try and interpret each component.
"""


# Plotting the principal components
plt.matshow(pd.np.transpose(app_pca_reduced.components_[0:5]), 
            cmap = 'Blues')

plt.xticks([0, 1, 2, 3, 4],
           ["PC 1", "PC 2", "PC 3", "PC 4", "PC 5"])

plt.colorbar()
plt.yticks(range(0, 75),
           app_df.columns[2:78],
           rotation=0,
           ha='left')

plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.savefig(graph_path + '2.Factor Loadings.png')
plt.show()


# Checking factor loadings
factor_loadings_df = pd.DataFrame(pd.np.transpose(app_pca_reduced.components_))
factor_loadings_df = factor_loadings_df.set_index(app_df.columns[2:77])

factor_loadings_df.to_excel(graph_path + '3.Factor Loadings.xlsx')


## Analyze factor strengths per user
X_pca_reduced = app_pca_reduced.transform(X_scaled_reduced)
X_pca_df = pd.DataFrame(X_pca_reduced)


## Rename your principal components and reattach demographic information
X_pca_df.columns = ['Society_disengagement', 
                    'Musical_Taste', 
                    'Tech_lag',
                    '21st_cent_professionalism',
                    'Mainstream_opposition']


final_pca_df = pd.concat([app_df.iloc[ : , 1 : 2] , app_df.iloc[ : , -11 : ] , X_pca_df], axis = 1)


## Analyze in more detail
# Creating folder in current directory to store graphs
cwd = os.getcwd()

if os.path.exists(cwd + '/2.2.Components by demographics'):
  shutil.rmtree(cwd + '/2.2.Components by demographics')
  
os.makedirs(cwd + '/2.2.Components by demographics')  
graph_path = cwd+'/2.2.Components by demographics/'


# Renaming age
age_names = {1 : '0-18',
             2 : '18-24',
             3 : '25-29',
             4 : '30-34',
             5 : '35-39',
             6 : '40-44',
             7 : '45-49',
             8 : '50-54',
             9 : '55-59',
             10 : '60-64',
             11 : '65-100'}

final_pca_df['q1'].replace(age_names, inplace = True)


# Renaming marital status
marital_names = {1 : 'Married',
                 2 : 'Single',
                 3 : 'Single with a partner',
                 4 : 'Separated/Widowed/Divorced'}

final_pca_df['q49'].replace(marital_names, inplace = True)


# Renaming household annual income
income_names = {1 : '$0-$10k',
                2 : '$10k-$15k',
                3 : '$15k-$20k',
                4 : '$20k-$30k',
                5 : '$30k-$40k',
                6 : '$40k-$50k',
                7 : '$50k-$60k',
                8 : '$60k-$70k',
                9 : '$70k-$80k',
                10 : '$80k-$90k',
                11 : '$90k-$100k',
                12 : '$100k-$125k',
                13 : '$125k-$150k',
                14 : '$150k-$1,000k'}

final_pca_df['q56'].replace(income_names, inplace = True)


# Analyzing by age
final_pca_df = final_pca_df.sort_values(by = ['q1'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
        sns.boxplot(x = 'q1',
                    y = X_pca_df[e],
                    data = final_pca_df,
                    ax = axes [i // 3][i % 3])

plt.tight_layout()
plt.savefig(graph_path+'1.Components by age.png')
plt.show()



# Analyzing by education
final_pca_df = final_pca_df.sort_values(by = ['q48'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
    sns.boxplot(x = 'q48',
                y =  X_pca_df[e],
                data = final_pca_df,
                ax = axes [i // 3][i % 3])

plt.tight_layout()
plt.savefig(graph_path+'2.Components by education.png')
plt.show()


# Analyzing by marital status
final_pca_df = final_pca_df.sort_values(by = ['q49'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
    sns.boxplot(x = 'q49',
                y =  X_pca_df[e],
                data = final_pca_df,
                ax = axes [i // 3][i % 3])

plt.tight_layout()
plt.savefig(graph_path+'3.Components by marital status.png')
plt.show()


# Analyzing by race
final_pca_df = final_pca_df.sort_values(by = ['q54'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
    sns.boxplot(x = 'q54',
                y =  X_pca_df[e],
                data = final_pca_df,
                ax = axes [i // 3][i % 3])

plt.tight_layout()
plt.savefig(graph_path+'4.Components by race.png')
plt.show()


# Analyzing by household annual income
final_pca_df = final_pca_df.sort_values(by = ['q56'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
    sns.boxplot(x = 'q56',
                y =  X_pca_df[e],
                data = final_pca_df,
                ax = axes [i // 3][i % 3])

plt.tight_layout()
plt.savefig(graph_path+'5.Components by income.png')
plt.show()


###############################################################################
# 2. PCA and clustering
###############################################################################

"""
Prof. Chase:
    That's right! We can combine both techniques! 
"""


## Analyze in more detail
# Creating folder in current directory to store graphs
cwd = os.getcwd()

if os.path.exists(cwd + '/2.3.Components by clusters'):
  shutil.rmtree(cwd + '/2.3.Components by clusters')
  
os.makedirs(cwd + '/2.3.Components by clusters')  
graph_path = cwd+'/2.3.Components by clusters/'


## Take your transformed dataframe
print(X_pca_df.head(n = 5))
print(pd.np.var(X_pca_df))


## Scale to get equal variance
scaler = StandardScaler()
scaler.fit(X_pca_df)
X_pca_clust = scaler.transform(X_pca_df)
X_pca_clust_df = pd.DataFrame(X_pca_clust)

print(pd.np.var(X_pca_clust_df))
X_pca_clust_df.columns = X_pca_df.columns


## Experiment with different numbers of clusters
app_k_pca = KMeans(n_clusters = 5,
                   random_state = 508)

app_k_pca.fit(X_pca_clust_df)
app_kmeans_pca = pd.DataFrame({'cluster': app_k_pca.labels_})


print(app_kmeans_pca.iloc[: , 0].value_counts())


## Analyze cluster centers
centroids_pca = app_k_pca.cluster_centers_
centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['Society_disengagement', 
                            'Musical_Taste', 
                            'Tech_lag',
                            '21st_cent_professionalism',
                            'Mainstream_opposition']

print(centroids_pca_df)


# Replacing cluster names
cluster_names = {0 : 'Traditional follower',
                 1 : 'Land of nowhere',
                 2 : 'Cool gamer',
                 3 : 'Superman consultant',
                 4 : '24/7 Metropoli hardworker'}

app_kmeans_pca['cluster'].replace(cluster_names, inplace = True)


# Sending data to Excel
centroids_pca_df.to_excel(graph_path+'1.KMeans centroids.xlsx')


## Analyze cluster memberships
clst_pca_df = pd.concat([app_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)

print(clst_pca_df.head())


## Reattach demographic information
final_pca_clust_df = pd.concat([app_df.iloc[ : , 1 : 2],
                                app_df.iloc[ : , -11 : ],
                                clst_pca_df],
                                axis = 1)

print(final_pca_clust_df.head(n = 5))


## Analyze in more detail 
# Renaming age
final_pca_clust_df['q1'].replace(age_names, inplace = True)


# Renaming marital status
final_pca_clust_df['q49'].replace(marital_names, inplace = True)


# Renaming household annual income
final_pca_clust_df['q56'].replace(income_names, inplace = True)


# Analyzing by age
final_pca_clust_df = final_pca_clust_df.sort_values(by = ['q1'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
        sns.boxplot(x = 'q1',
                    y = X_pca_df[e],
                    hue = 'cluster',
                    data = final_pca_clust_df,
                    ax = axes [i // 3][i % 3])

for c in axes:
    for ax in c:
        ax.axhline(0, color = 'w')
        
plt.tight_layout()
plt.savefig(graph_path+'2.PCs by age.png')
plt.show()



# Analyzing by education
final_pca_clust_df = final_pca_clust_df.sort_values(by = ['q48'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
    sns.boxplot(x = 'q48',
                y =  X_pca_df[e],
                hue = 'cluster',
                data = final_pca_clust_df,
                ax = axes [i // 3][i % 3])

for c in axes:
    for ax in c:
        ax.axhline(0, color = 'w')
        
plt.tight_layout()
plt.savefig(graph_path+'3.PCs by education.png')
plt.show()


# Analyzing by marital status
final_pca_clust_df = final_pca_clust_df.sort_values(by = ['q49'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
    sns.boxplot(x = 'q49',
                y =  X_pca_df[e],
                hue = 'cluster',                
                data = final_pca_clust_df,
                ax = axes [i // 3][i % 3])

for c in axes:
    for ax in c:
        ax.axhline(0, color = 'w')
        
plt.tight_layout()
plt.savefig(graph_path+'4.PCs by marital status.png')
plt.show()


# Analyzing by race
final_pca_clust_df = final_pca_clust_df.sort_values(by = ['q54'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
    sns.boxplot(x = 'q54',
                y =  X_pca_df[e],
                hue = 'cluster',                
                data = final_pca_clust_df,
                ax = axes [i // 3][i % 3])

for c in axes:
    for ax in c:
        ax.axhline(0, color = 'w')
        
plt.tight_layout()
plt.savefig(graph_path+'5.PCs by race.png')
plt.show()


# Analyzing by household annual income
final_pca_clust_df = final_pca_clust_df.sort_values(by = ['q56'])

fig, axes = plt.subplots(2, 3, figsize = (16, 8))
for i, e in enumerate(X_pca_df.columns):
    sns.boxplot(x = 'q56',
                y =  X_pca_df[e],
                hue = 'cluster',                
                data = final_pca_clust_df,
                ax = axes [i // 3][i % 3])

for c in axes:
    for ax in c:
        ax.axhline(0, color = 'w')
        

plt.tight_layout()
plt.savefig(graph_path+'6.PCs by income.png')
plt.show()

