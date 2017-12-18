# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans
from utils import pd_load

# data path config
parent_path = 'F:\\毕业论文\\experiments_thesis\\'
trainfile_path = 'datasets\\birds\\birds-train.csv'
testfile_path = 'datasets\\birds\\birds-test.csv'
labels_num = 19

# data load
traindata = pd_load(parent_path + trainfile_path)
testdata = pd_load(parent_path + testfile_path)

all_data = traindata.append(
    testdata,
    ignore_index=True)  # merge traindata and test data for clustering

all_data_labels = all_data.iloc[:, -labels_num:]

# para relative config
random_labels_num = 5
randomlist = random.sample(range(19), random_labels_num)
index_list = pd.indexes.range.RangeIndex(0)
for id in randomlist:
    index_list = index_list.union(
        all_data[all_data.iloc[:, id - labels_num] == 1].index)

index_cluster = all_data.index.difference(index_list)
index_stream = index_list

cluster_data = all_data.iloc[index_cluster]
stream_data = all_data.iloc[index_stream]

# cluster model
num_cluster = 8
