#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 12:25:33 2022

@author: majid
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

class Recommender():
    """
    This object has 4 main item recommenders which are :
        cluster_recom
        item_recome
        find_common
        item2user_recom
    """
    def __init__(self, item, user, df, df_cluster, cos_sim_data, cos_sim_clusters, Unique_pca_data):
        """
        item: the searched item by the user
        user: the customerID
        df: the main dataset [CustomerID, Itemname]
        df_cluster: a datafram with [CustomerID, PCA, Grouped_items] columns 
        cos_sim_data: cosine similarity between all items
        cos_sim_clusters: a dataframe containing the cosine similarity between clusters of users
        Unique_pca_data: the pca of Unique items
        """
        self.item = item
        self.user = user
        self.df = df
        self.df_cluster = df_cluster
        self.df_unique = pd.DataFrame(pd.unique(df.Itemname),index=None,columns=['Itemname'])
        self.cos_sim_clusters = cos_sim_clusters
        self.cos_sim_data = cos_sim_data
        self.items = []
        self.Unique_pca_data = Unique_pca_data
        
    def cluster_recom(self):
        """
        finds the nearest user to the target user and recommends the items that
        the other user has bought
        Note: we might need to ignore the common items between these 2 users
        """
        self.idx = self.df_cluster.index[self.df_cluster.CustomerID == self.user]
        self.index_recomm = self.cos_sim_clusters[str(self.idx[0])].sort_values(ascending=False).index.tolist()[1]
        self.item_recomm =  self.df_cluster['Grouped_items'].loc[self.index_recomm]
        return self.item_recomm
    
    def item_recom(self):
        """
        The most similar items to the target item
        Note: we might need to ignore the items that the user has already bought
        """
        self.idx = self.df_unique.index[pd.unique(self.df_unique.Itemname) == self.item]
        self.index_recomm = self.cos_sim_data[str(self.idx[0])].sort_values(ascending=False).index.tolist()[1:6]
        self.item_recomm =  self.df_unique['Itemname'].loc[self.index_recomm].values
        return self.item_recomm
    
    def find_users(self):
        """
        finds the users that bought the same item that the target user searches for
        it will return the list of items that they bought and the index of these users
        """
        founded_users_index = []
        founded_items = []
        for i in range(self.df_cluster.count()[0]):
            if self.item in self.df_cluster.Grouped_items.iloc[i]:
                if self.df_cluster.CustomerID.iloc[i] != self.user:
                    founded_users_index.append(i)
                    founded_items.append(self.df_cluster.Grouped_items.iloc[i])
        return founded_items, founded_users_index
    
    def most_frequent(List):
            return max(set(List), key = List.count)
    
    def find_common(self):
        """
        finds the most frequent items among the users who have bought the target item
        """
        items = []
        founded_items, founded_users_index = self.find_users()
        for l in founded_items:
            items += l
        d = Counter(items)
        final_list = [x[0] for x in d.most_common()[1:6]]
        return final_list
    
 
        
    def user_PCA(self):
        """
        find the PCA of the target user
        """
        self.pca_cluster = self.df_cluster.PCA[self.df_cluster.CustomerID == self.user]
        return self.pca_cluster
    
    def item2user_recom(self):
        """
        calculates the similarity of the items which are not in the target user's basket
        to the target user and recommends the most similar ones
        """
        # cosine_smiliarity takes ndarrays as input so we need to convert our data to this type
        n_comp = 10
        self.Unique_pca_array = np.ndarray((self.Unique_pca_data.count()[0],n_comp))
        for i,l in enumerate(np.asarray(self.Unique_pca_data.iloc[:])):
            self.Unique_pca_array[i] = np.asarray(l[1:11])
        self.cos_sim = pd.DataFrame(cosine_similarity(np.asarray(self.user_PCA().iloc[0]).reshape(1, 10), 
                                                      self.Unique_pca_array))
        self.idx_recom = self.cos_sim.iloc[0].sort_values(ascending=False).index.tolist()[0:5]
        items= []
        # checking whether the items are already in the user's basket or not
        for i in self.idx_recom:
            if pd.unique(self.df.Itemname)[i] not in self.df_cluster.Grouped_items[self.df_cluster.CustomerID == self.user]:
                items.append(pd.unique(self.df.Itemname)[i])
                # returns 5 items
                if len(items)>5:
                    break
        return items
        
        