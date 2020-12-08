import os
import json
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, tf.Tensor):
            return obj.numpy().tolist()
        else:
            return super(NpEncoder, self).default(obj)

def saveDict(dictionary,folder,data_name):
    folder_path = os.path.join(str(folder), str(data_name))
    #folder_path = folder + '/' + data_name
    string=json.dumps(dictionary,indent=2,cls=NpEncoder)
    with open(folder_path,'w') as f:
        f.write(string)
    f.close()

def loadDict(folder,data_name):
    folder_path = os.path.join(str(folder), str(data_name))
    string = open(folder_path)
    return json.load(string)

def reverseDict(dictionary):
    return dict(zip(dictionary.values(), dictionary.keys()))

def printDict(dictionary):
    for key, value in dictionary.items():
        print(key,':',value)


def createDictBydf(df, column, isSort = False):
    df_column = df[column]
    if isSort:
        df_column = df_column.sort_values()
    list_column = df_column.unique()
    list_column = list(zip(np.unique(list_column, return_inverse = True)[0], np.unique(list_column, return_inverse = True)[1]))
    dict_column = {}
    for k,v in enumerate(list_column):
        dict_column[int(v[1])] = v[0]
    return dict_column


def prepareFolder(path):
    """Create folder from path."""
    if not os.path.isdir(path):
        os.makedirs(path)
    return path

def transferStrT2Seconds(timestamps):
    df = pd.DataFrame({'timestamp':timestamps})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    minTimestamp = df['timestamp'].min()
    df['timestamp'] = df['timestamp'] - df['timestamp'].min()
    df['timestamp'] = df['timestamp'].apply(lambda x: x.total_seconds()).astype(np.int64)
    df.sort_values(by = 'timestamp', inplace = True)
    return list(np.array(df['timestamp'])), minTimestamp

def transferStrT2Dir(timeLC):
    a = timeLC[0]
    b = timeLC[1]
    return [a.replace(':','-'), b.replace(':','-')]


def getLegend(params, keyNumLimit = 1):
    """Create save legend from params dictionary"""
    legend = ''
    for key,value in params.items():
        if isinstance(value, list):
            legend = legend + '_' + key[0:keyNumLimit].upper() + '_['
            flag = 0
            for item in value:
                if item == 1e9:
                    if flag == 0:
                        legend = legend + '1e9'
                    else:
                        legend = legend + ',1e9'
                else:
                    if flag == 0:
                        legend = legend + str(item)
                    else:
                        legend = legend + ',' + str(item)
                flag += 1 
            legend = legend + ']'
        else:
            legend=legend+'_'+key[0:keyNumLimit].upper()+'_'+str(value)

    return legend

def save_as_pkl(save_path, data):
    fw = open(save_path, "wb")
    pickle.dump(data, fw)
    fw.close()

def load_pkl(save_path):
    fw = open(save_path, 'rb')
    data = pickle.load(fw)
    fw.close()
    return data