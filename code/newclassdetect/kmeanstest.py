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
randomlist=random.sample(range(19),random_labels_num)


