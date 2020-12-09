import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from utils.this_queue import OurQueue
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from public import *
from KDDCupDataProcessor import _KDDCupDataProcessor
from OJDataProcessor import _OJDataProcessor

class _DataProcessor:
    # minTimestamp必须传值，默认值不管用
    def __init__(self, userLC, problemLC, timeLC, dataType = 'kdd', TmpDir = './data/'):
        if dataType=='kdd':
            self.dataprocessor= _KDDCupDataProcessor(userLC, problemLC, timeLC, TmpDir = TmpDir)
        elif dataType=='oj':
            self.dataprocessor= _OJDataProcessor(userLC, problemLC, timeLC, TmpDir = TmpDir)

        self.datasetName= self.dataprocessor.datasetName
        self.TmpDir = self.dataprocessor.TmpDir
        self.LC_params = self.dataprocessor.LC_params
        self.LCDataDir = os.path.join(self.TmpDir, 'LCData',self.datasetName+getLegend(self.LC_params))

    # 存储的是学生五折交叉验证的划分结果
    def loadSplitInfo(self, kFold):
        if os.path.exists(os.path.join(self.LCDataDir, 'splitedInformation_kFold('+str(kFold)+').json')):
            dict_data = loadDict(self.LCDataDir, 'splitedInformation_kFold('+str(kFold)+').json')
            return dict_data
        [df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
        all_users = df['user_id'].unique()

        dict_data = {}

        kf = KFold(n_splits = kFold, shuffle = True)
        splits = kf.split(all_users)

        for run_id, (i_user_train, i_user_test) in enumerate(splits):
            users_train = all_users[i_user_train]
            users_test = all_users[i_user_test]
            dict_data[run_id]={'train':users_train,'test':users_test}

        saveDict(dict_data,self.LCDataDir, 'splitedInformation_kFold('+str(kFold)+').json')
        return dict_data

    def loadDKTbatchData(self, trainRate, batch_size):
        [df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
        df = df.drop(['timestamp'],axis=1)
        item_num = StaticInformation['itemNum']
        df['skill_id_with_correct'] = df['correct'] * item_num + df['item_id']

        data = df.groupby('user_id').apply(
            lambda r: (r['skill_id_with_correct'].values[:-1], r['item_id'].values[1:], r['correct'].values[1:])
        )
        i = 0
        users = []
        items = []
        corrects = []
        for it in data:
            if i%100 == 0:
                print(i,'/',len(df))
            i += 1
            users.append(list(it[0]))
            items.append(list(it[1]))
            corrects.append(list(it[2]))

        def __to_dataset(data):
            data = tf.ragged.constant(data)
            data = tf.data.Dataset.from_tensor_slices(data)
            data = data.map(lambda x: x)
            return data

        users = __to_dataset(users)
        items = __to_dataset(items)
        corrects = __to_dataset(corrects)

        dataset = tf.data.Dataset.zip((users, items, corrects))
        # dtype
        dataset = dataset.map(lambda inputs, data, label: (tf.cast(inputs, dtype=tf.int32), tf.cast(data, dtype=tf.float32), tf.cast(label, dtype=tf.float32))
                              )
        # dim
        dataset = dataset.map(lambda inputs, data, label: (inputs, tf.expand_dims(data, axis=-1), tf.expand_dims(label, axis=-1))
                              )
        # concat
        dataset = dataset.map(lambda inputs, data, label: (inputs, tf.concat([data, label], axis=-1))
                              )

        def __split_dataset(dataset, trainRate):
            if tf.__version__ == '2.2.0':
                data_size = tf.data.experimental.cardinality(dataset).numpy()
            else:
                data_size = dataset.cardinality().numpy()
            train_size = int(data_size * trainRate)

            train_dataset = dataset.take(train_size)
            test_dataset = dataset.skip(train_size)
            return train_dataset, test_dataset

        train_dataset, test_dataset = __split_dataset(dataset, trainRate)
        train_dataset = train_dataset.padded_batch(batch_size, drop_remainder=False)
        test_dataset = test_dataset.padded_batch(batch_size, drop_remainder=False)
        return train_dataset, test_dataset, item_num

    def loadDKVMNbatchData(self, trainRate, batch_size):
        [df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
        df = df.drop(['timestamp'],axis=1)
        item_num = StaticInformation['itemNum']
        df['skill_id_with_correct'] = df['correct'] * item_num + df['item_id']

        data = df.groupby('user_id').apply(
            lambda r: (r['skill_id_with_correct'].values,
                       r['item_id'].values,
                       r['correct'].values)
        )
        i = 0
        users = []
        items = []
        corrects = []
        for it in data:
            if i%100 == 0:
                print(i,'/',len(df))
            i += 1
            users.append(list(it[0]))
            items.append(list(it[1]))
            corrects.append(list(it[2]))

        def __to_dataset(data):
            data = tf.ragged.constant(data)
            data = tf.data.Dataset.from_tensor_slices(data)
            data = data.map(lambda x: x)
            return data

        users = __to_dataset(users)
        items = __to_dataset(items)
        corrects = __to_dataset(corrects)

        dataset = tf.data.Dataset.zip((users, items, corrects))
        # dtype
        dataset = dataset.map(lambda inputs, data, label: (tf.cast(inputs, dtype=tf.int32), tf.cast(data, dtype=tf.int32), tf.cast(label, dtype=tf.float32))
                              )
        # dim
        dataset = dataset.map(lambda inputs, data, label: (tf.expand_dims(inputs, axis=-1), tf.expand_dims(data, axis=-1), tf.expand_dims(label, axis=-1))
                              )
        # concat
        dataset = dataset.map(lambda inputs, data, label: (tf.concat([data, inputs], axis=-1), label)
                              )

        def __split_dataset(dataset, trainRate):
            if tf.__version__ == '2.2.0':
                data_size = tf.data.experimental.cardinality(dataset).numpy()
            else:
                data_size = dataset.cardinality().numpy()
            train_size = int(data_size * trainRate)

            train_dataset = dataset.take(train_size)
            test_dataset = dataset.skip(train_size)
            return train_dataset, test_dataset

        train_dataset, test_dataset = __split_dataset(dataset, trainRate)
        train_dataset = train_dataset.padded_batch(batch_size, drop_remainder=False)
        test_dataset = test_dataset.padded_batch(batch_size, drop_remainder=False)
        return train_dataset, test_dataset, item_num
    # all_features = ['lasttimes', 'skills', 'attempts', 'wins']
    # wins没写呢
    def loadSparseDF(self, active_features = ['lasttimes', 'skills', 'attempts'], window_lengths = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]):
        if os.path.exists(os.path.join(self.LCDataDir, 'sparse_df.npz')):
            sparse_df = sparse.csr_matrix(sparse.load_npz('sparse_df.npz'))
            return sparse_df

        [df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()

        NB_OF_TIME_WINDOWS = len(window_lengths)

        # Transform q-matrix into dictionary
        dict_q_mat = {i:set() for i in range(QMatrix.shape[0])}
        for elt in np.argwhere(QMatrix == 1):
            dict_q_mat[elt[0]].add(elt[1])

        X={}
        X['df'] = np.empty((0,4)) # Keep track of the original dataset
        if 'skills' in active_features:
            X['skills'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
        if 'attempts' in active_features or 'lasttimes' in active_features:
            X['lasttimes'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
            X['deltaT'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
            X['attempts'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))
        if 'wins' in active_features:
            X['wins'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))

        q = defaultdict(lambda: OurQueue(window_lengths))  # Prepare counters for time windows

        for stud_id in df['user_id'].unique():
            df_stud = df[df['user_id']==stud_id][['user_id', 'item_id', 'timestamp', 'correct']].copy()
            df_stud.sort_values(by='timestamp', inplace=True) # Sort values 
            df_stud = np.array(df_stud)

            X['df'] = np.vstack((X['df'], df_stud)) #rawdata 0-4列

            #skills
            if 'skills' in active_features:
                skills_temp = QMatrix[df_stud[:,1].astype(int)].copy()
                X['skills'] = sparse.vstack([X['skills'],sparse.csr_matrix(skills_temp)])


            #lasttimes and attempts
            if 'attempts' in active_features or 'lasttimes' in active_features:
                attempts = np.zeros((df_stud.shape[0], QMatrix.shape[1] * NB_OF_TIME_WINDOWS))
                lasttimes = np.zeros((df_stud.shape[0], QMatrix.shape[1]))
                deltaT = np.zeros((df_stud.shape[0], QMatrix.shape[1]))
                for l, (item_id, t) in enumerate(zip(df_stud[:,1], df_stud[:,2])):
                    for skill_id in dict_q_mat[item_id]:
                        attempts[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.array(q[stud_id, skill_id].get_counters(t))
                        #print(attempts[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS])
                        #print(l, skill_id*NB_OF_TIME_WINDOWS,(skill_id+1)*NB_OF_TIME_WINDOWS)
                        lastT = q[stud_id, skill_id].get_last()
                        lasttimes[l, skill_id] = lastT
                        deltaT[l, skill_id] = t-lastT
                        q[stud_id, skill_id].push(t)
                X['attempts'] = sparse.vstack([X['attempts'],sparse.csr_matrix(attempts)])
                X['lasttimes'] = sparse.vstack([X['lasttimes'],sparse.csr_matrix(lasttimes)])
                X['deltaT'] = sparse.vstack([X['deltaT'],sparse.csr_matrix(deltaT)])

        sparse_df = sparse.hstack([sparse.csr_matrix(X['df']),sparse.hstack([X[agent] for agent in active_features])]).tocsr()
        sparse.save_npz(os.path.join(self.LCDataDir, 'sparse_df'+str(window_lengths)+'.npz'), sparse_df)
        return sparse_df





# oj
'''
userLC = [10,500,0.1,1]
problemLC = [10,500,0,1]
#hdu原始数据里的最值，可以注释，不要删
low_time = "2018-06-01 00:00:00" 
high_time = "2018-11-29 00:00:00"
timeLC = [low_time, high_time]
a = _DataProcessor(userLC, problemLC, timeLC, 'oj')
print('**************LC_params**************')
printDict(a.LC_params)
features = ['skills', 'attempts']
window_lengths = [1,3600 * 1e19]
#window_lengths = [3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
sparse_df = a.loadSparseDF(features, window_lengths)
print('**************sparse_df**************')
print(sparse_df.shape)
sparse_df = pd.DataFrame(sparse_df).head(1000)
sparse_df.to_csv(os.path.join(a.LCDataDir,'test.txt'), sep='\t', index=False)
'''

'''
[df, QMatrix, StaticInformation, DictList] = a.dataprocessor.loadLCData()
print('**************StaticInformation**************')
printDict(StaticInformation)
SplitInfo = a.loadSplitInfo(5)
print('**************SplitInfo**************')
printDict(SplitInfo[list(SplitInfo.keys())[0]])
[train, test, itemNUm] = a.loadDKTbatchData(0.8, 256)
print('**************dataset**************')
print(type(train))
print(type(test))
'''




