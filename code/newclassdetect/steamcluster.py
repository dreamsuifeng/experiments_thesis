# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class steamcluster:

    def __init__(self,num_cluster):
        self.num_cluster=num_cluster
        self.centers_list=[]
        self.centers_