# -*- coding:utf8 -*_

import numpy as np
import pandas as pd
import arff


def load_arff(filepath):
    '''
    Args:
        filepath: the path of data file path with arff type
    Return:
        A dict data contains arff data
    '''
    return arff.load(open(filepath, 'r'))


def to_csv(arffpath, to_csv_path, num_labels):
    '''
    Transfer the arff format to csv format
    Args:
        arffpath: arff data path
        to_csv_path: store the data file path with csv format
        num_labels: number of labels
    Return:
        output the file which store the arrfpath
    '''
    arff_data = load_arff(arffpath)
    attris_names = arff_data['attributes']
    list_attris = []
    for a in attris_names:
        list_attris.append(a[0])
    datas = np.asarray(arff_data['data'])
    datas_pd = pd.DataFrame(datas, columns=list_attris)
    print(datas_pd.shape)
    datas_pd.to_csv(to_csv_path, header=True, index=False)


def pd_load(filepath, index_is=None):
    '''
    use pandas to load data with csv file
    Args:
        filepath: data file path
    Return:
        return data with pandas data structure
    '''
    return pd.read_csv(filepath, index_col=index_is)


if __name__ == '__main__':
    arffpath = 'F:/毕业论文/experiments_thesis/datasets/birds/birds-test.arff'
    outputpath = 'F:\\毕业论文\\experiments_thesis\\datasets\\birds\\birds-test.csv'
    to_csv(arffpath, outputpath, 19)
