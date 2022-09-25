#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 10:43:08 2022

@author: majid
"""

from Recommenders import Recommender
import pandas as pd
import numpy as np

# Data Load
df = pd.read_excel('data/Assignment-1_Data.xlsx')
df.drop(['BillNo', 'Date'], axis=1, inplace=True)
# droping the rows that have NaN itemname
df.dropna(subset=['Itemname'] ,inplace=True)
# bfill: fills the NaN values with privious row
df.CustomerID.fillna(method="bfill", inplace=True)

df_cluster = pd.read_json('data/df_cluster.json')

Unique_pca_data = pd.read_csv('data/Unique item PCA.csv')

# cos similariries load
cos_sim_data = pd.read_csv('data/cos_sim_data.csv')

cos_sim_clusters = pd.read_csv('data/cos_sim_clusters.csv')


# main loop to get item recommendations and preventing data loading for each search
con = 'y'
while (con =='y'):
    # inputs
    item = input('Enter the item: ') # 'POPPY'S PLAYHOUSE KITCHEN'
    user = float(input('Enter the user ID: ')) # 17341
    
    # recommender
    recoms = Recommender(item=item, 
                         user=user, 
                         df=df, 
                         df_cluster = df_cluster, 
                         cos_sim_data = cos_sim_data, 
                         cos_sim_clusters = cos_sim_clusters,
                         Unique_pca_data = Unique_pca_data)
    
    # printing results
    print("----------------------------------------")
    print('The most similar items to the target items are: ')
    recoms1 = recoms.item_recom()
    for i,item in enumerate(recoms1):
        print(f"{i+1}- {item}")

    print("----------------------------------------")
    print('Some items that the nearest user to the target user has bought: ')
    recoms2 = recoms.cluster_recom()[0:5]
    for i,item in enumerate(recoms2):
        print(f"{i+1}- {item}")

    print("----------------------------------------")
    print('Most common items among the users who have bought this item:')
    recoms3 = recoms.find_common()
    for i,item in enumerate(recoms3):
        print(f"{i+1}- {item}")
  
    print("----------------------------------------")
    print('Nearest Items to the target user:')
    recoms4 = recoms.item2user_recom()
    for i,item in enumerate(recoms4):
        print(f"{i+1}- {item}")
        
    print("----------------------------------------")
    
    # check whether there are more items to be searched or not
    con = input('Do you want to continue? \'y\' or \'n\': ').lower()


