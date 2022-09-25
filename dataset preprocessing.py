#!/usr/bin/env python
# coding: utf-8

# In[1]: import pachages


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# In[2]: Items embeddings and cosine similarities


"""
BillNo: 6-digit number assigned to each transaction. Nominal.
Itemname: Product name. Nominal.
Quantity: The quantities of each product per transaction. Numeric.
Date: The day and time when each transaction was generated. Numeric.
Price: Product price. Numeric.
CustomerID: 5-digit number assigned to each customer. Nominal.
Country: Name of the country where each customer resides. Nominal.
"""
df = pd.read_excel('data/Assignment-1_Data.xlsx')
# these 2 columns won't provide any useful information, so we drop them
df.drop(['BillNo', 'Date'], axis=1, inplace=True)
df.head(10)



# Nan values problem addressing
# For Itemname we removed the rows containg NaN <br> For CustomerID we replaced NaNs with previous values
for key in df.keys():
    print(f"{key} has {df[key].isna().sum()} NaN values")


df.dropna(subset=['Itemname'] ,inplace=True)
# bfill: fills the Nan values with privious row
df.CustomerID.fillna(method="bfill", inplace=True)
print("\nAfter addressing NaN values:")
for key in df.keys():
    print(f"{key} has {df[key].isna().sum()} NaN values")


# print some description of the data
print("\n")
print(df.info())
print(f"Data Desciption: \n {df.describe()}")
print(f"Number of unique customer IDs: {len(pd.unique(df.CustomerID))}")
print(f"Number of unique Items: {len(pd.unique(df.Itemname))}")


# Itemname column embedding

model = SentenceTransformer('all-MiniLM-L6-v2') # alternative model: distilbert-base-nli-mean-tokens
embeddings = model.encode(pd.unique(df.Itemname).tolist(), show_progress_bar=False)


# saving the numpy array contains the instances of items(uniquely)
# it has 4185 rows (unique items)

#np.save('data/unique items embeddings.npy', embeddings)
#print("Unique Items embeddings saved")

# calculating the PCA of Unique items embeddings
n_comp = 10 # the desired dimension
pca = PCA(n_components=n_comp)
pca.fit(embeddings)
Unique_pca_data = pd.DataFrame(pca.transform(embeddings))
Unique_pca_data.to_csv('data/Unique item PCA.csv')
print("Unique item PCA saved")

# calculating the cross table including cosine similarity distances
from sklearn.metrics.pairwise import cosine_similarity
cos_sim_data = pd.DataFrame(cosine_similarity(embeddings))

n=10
print(f"{n} rows of Cosine Similarity of unique items: \n {cos_sim_data.head(n)}")



# saving data (so there is no need to run embedding and calculating of cos_sim_data cells)
cos_sim_data.to_csv('data/cos_sim_data.csv', header=True, index=False)
print("Cosine Similarities of Unique items saved")


# In[2]: clustering the data for each user

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df.Itemname.tolist(), show_progress_bar=True)
#np.save('items embeddings.npy', embeddings)

# PCA of data to decrease the dimensions and space usage

#X = np.array(embeddings)
pca = PCA(n_components=n_comp)
pca.fit(embeddings)
pca_data = pd.DataFrame(pca.transform(embeddings))
print(f"PCA of items: \n {pca_data.head()}")

# grouping - caution: groupby sorts the values
df['PCA'] = pca_data.to_numpy().tolist()
df_cluster = pd.DataFrame(columns=['mean', 'grouped_items', 'CustomerID']).sum()
df_cluster = df[["PCA", "CustomerID"]].groupby(['CustomerID'],as_index=False).sum() #concats the PCAs of rows
df_cluster = pd.concat([df_cluster, 
                        pd.DataFrame(df[["Itemname", "CustomerID"]].groupby(['CustomerID'])['Itemname'].apply(list)).reset_index(level=None, drop=False, inplace=False)['Itemname']],
                       axis=1)
df_cluster.columns = ['CustomerID', 'PCA', 'Grouped_items']
print(f"The dataframe that shows clusters related to each user: \n {df_cluster.head()}")

# .sum method concatenates the PCAs in a single list so the shape would be [n*n_comp]
# we need to reshape these lists to [n, n_comp] and then getting the mean of them
for i in range(df_cluster.count()[0]):
    df_cluster.PCA.iloc[i] = np.mean(np.array([df_cluster.PCA.iloc[i][j: j+n_comp] for j in range(0, len(df_cluster.PCA.iloc[i]), n_comp)]),axis=0)

# to_csv converts list to string so we use to_json
df_cluster.to_json('data/df_cluster.json')
print("df_cluster saved")

# calculating the cross table including cosine similarity distances of clusters
# the cosine_similarity takes ndarray of arrays so we transform our data to that type
a = np.ndarray((df_cluster.PCA.count(),n_comp))
for i,l in enumerate(df_cluster.PCA.iloc[:]):
    a[i] = np.asarray(l)
cos_sim_clusters = pd.DataFrame(np.array(cosine_similarity(a)))
cos_sim_clusters.to_csv('data/cos_sim_clusters.csv')
print("cosine similarities of clusters saved")


