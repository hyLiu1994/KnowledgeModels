import os
import re
import sys
import json
import time
import math
import pickle
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import KFold
from public import *
# coding = utf-8

class _OJDataProcessor(object):
    def __init__(self, userLC, problemLC, timeLC, OnlyRight = True, drop_duplicates = True, datasetName = 'hdu',  TmpDir = "./data/"):
        self.datasetName = datasetName

        #hduOJ专用数据预处理
        #userLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
        #ProblemLC = [最少提交次数，最多提交次数，最低通过率，最高通过率]
        #timeLC = [起始时间（单位秒），终止时间（秒）]
        # 当OnlyRight为真的时候，只考虑Accepted，其它所有情况划分为一类，等OnlyRight为假的时候
        # 分为这几种情况dic_status = {'Accepted': 1, 'Wrong Answer': 0, 'Time Limit Exceeded': 2, 'other': 3}
    
        self.LC_params={}
        self.LC_params['userLC'] = userLC
        self.LC_params['problemLC'] = problemLC
        self.LC_params['timeLC'] = transferStrT2Dir(timeLC)
        self.LC_params['Right'] = OnlyRight
        self.LC_params['dropDup'] = drop_duplicates

        self.TmpDir = TmpDir
        self.RawDataDir = os.path.join(self.TmpDir, 'rawData', self.datasetName)
        self.RawDataName = 'hdu_RawSubmitRecord.txt'
        self.RawKnowledge2Problem = 'hdu_RawKnowledge2Problem.txt'
        self.LCDataDir = os.path.join(self.TmpDir, 'LCData',self.datasetName+getLegend(self.LC_params))


        self.timeLC, self.minTimestamp = transferStrT2Seconds(timeLC)

    def loadLCData(self):
        # 所有的字典都是id2str

        flag = 0
        if not os.path.exists(os.path.join(self.LCDataDir, 'studentSubmitRecords.csv')):
            flag = 1
        if not os.path.exists(os.path.join(self.LCDataDir, 'QMatrix.npz')):
            flag = 1
        if not os.path.exists(os.path.join(self.LCDataDir, 'StaticInformation.json')):
            flag = 1
        if not os.path.exists(os.path.join(self.LCDataDir, 'userInformation.json')):
            flag = 1
        if not os.path.exists(os.path.join(self.LCDataDir, 'itemInformation.json')):
            flag = 1
        if not os.path.exists(os.path.join(self.LCDataDir, 'knowledgeInformation.json')):
            flag = 1

        if flag == 0:
            df = pd.read_csv(os.path.join(self.LCDataDir, 'studentSubmitRecords.csv'), sep='\t')
            QMatrix = sparse.load_npz(os.path.join(self.LCDataDir,'QMatrix.npz'))
            StaticInformation = loadDict(self.LCDataDir,'StaticInformation.json')
            userInformation = loadDict(self.LCDataDir,'userInformation.json')
            itemInformation = loadDict(self.LCDataDir,'itemInformation.json')
            knowledgeInformation = loadDict(self.LCDataDir,'knowledgeInformation.json')
            DictList = [userInformation, itemInformation, knowledgeInformation]
            return df, QMatrix, StaticInformation, DictList

        prepareFolder(self.LCDataDir)
        df = pd.read_csv(os.path.join(self.RawDataDir, self.RawDataName), header = None, delimiter='   ', engine='python')
        df.columns = ['0','1','2','3','4','5','6','7','8']
        df = df[['8','3','1','2']]
        df.rename(columns={'8':'user_id', '3':'item_id', '1':'timestamp', '2':'correct'}, inplace=True) 

        dic_status = {'Accepted': 1, 'Wrong Answer': 0, 'Time Limit Exceeded': 2, 'other': 3}

        if self.LC_params['Right']:
            df['correct'].loc[df['correct'] == 'Accepted'] = 1
            df['correct'].loc[df['correct'] != 1] = 0
        else:
            df['correct'].loc[df['correct'] == 'Accepted'] = 1
            df['correct'].loc[df['correct'] == 'Wrong Answer'] = 0
            df['correct'].loc[df['correct'] == 'Time Limit Exceeded'] = 2
            df['correct'].loc[(df['correct'] != 0) & (df['correct'] != 1) & (df['correct'] != 2)] = 3


        # 1 timeLC
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'] - self.minTimestamp
        #df['timestamp'] = df['timestamp'].apply(lambda x: x.total_seconds() / (3600*24))
        df['timestamp'] = df['timestamp'].apply(lambda x: x.total_seconds()).astype(np.int64)
        #print(df['timestamp'].max(),self.LC_params['timeLC'][0],self.LC_params['timeLC'][1])
        df = df[df.timestamp >= self.timeLC[0]]
        df = df[df.timestamp <= self.timeLC[1]]

        # 2 userLC
        df = df.groupby('user_id').filter(lambda x: len(x) >= self.LC_params['userLC'][0])
        df = df.groupby('user_id').filter(lambda x: len(x) <= self.LC_params['userLC'][1])
        df = df.groupby('user_id').filter(lambda x: len(x['correct']==1)/len(x) >= self.LC_params['userLC'][2])
        df = df.groupby('user_id').filter(lambda x: len(x['correct']==1)/len(x) <= self.LC_params['userLC'][3])

        # 3 problemLC
        df = df.groupby('item_id').filter(lambda x: len(x) >= self.LC_params['problemLC'][0])
        df = df.groupby('item_id').filter(lambda x: len(x) <= self.LC_params['problemLC'][1])
        df = df.groupby('item_id').filter(lambda x: len(x['correct']==1)/len(x) >= self.LC_params['problemLC'][2])
        df = df.groupby('item_id').filter(lambda x: len(x['correct']==1)/len(x) <= self.LC_params['problemLC'][3])

        df.sort_values(by = 'timestamp', inplace = True)
        df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes
        df['correct'] = df['correct'].astype(np.int32) # Cast outcome as int32

        userInformation = createDictBydf(df, 'user_id')
        itemInformation = createDictBydf(df, 'item_id')
        user2id = reverseDict(userInformation)
        item2id = reverseDict(itemInformation)


        # Transform ids into numeric
        df['user_id'] = df['user_id'].apply(lambda x: user2id[x])
        df['item_id'] = df['item_id'].apply(lambda x: item2id[x])

        # Build Q-matrix
        knowledgeId = 0
        QMatrix = []
        knowledgeInformation = {}
        numKCs = 0
        with open(os.path.join(self.RawDataDir, self.RawKnowledge2Problem), 'r', encoding = 'UTF-8') as f:
            for line in f: 
                lineArray = line.split(':')
                lineArray[1] = lineArray[1].split(',')

                if (lineArray[0] not in knowledgeInformation.values()):
                    knowledgeInformation[knowledgeId] = lineArray[0]
                    knowledgeId += 1
                QMatrix.append([0 for j in range(len(df['item_id'].unique()))])
                for i in range(0,len(lineArray[1])):
                    if (int(lineArray[1][i]) in item2id.keys()):
                        QMatrix[-1][item2id[int(lineArray[1][i])]] = 1
                        numKCs += 1

        QMatrix = np.array(QMatrix).T

        StaticInformation = {}
        StaticInformation['userNum'] = len(df['user_id'].unique())
        StaticInformation['itemNum'] = len(df['item_id'].unique())
        StaticInformation['knowledgeNum'] = QMatrix.shape[1]
        StaticInformation['recordsNum'] = df.shape[0]

        StaticInformation['aveUserSubmit'] = df.shape[0] / len(df['user_id'].unique())
        StaticInformation['aveitemNumSubmit'] = df.shape[0] / len(df['item_id'].unique())
        StaticInformation['aveItemContainKnowledge'] = numKCs / len(df['item_id'].unique())

        DictList = [userInformation, itemInformation, knowledgeInformation]

        # Save data
        sparse.save_npz(os.path.join(self.LCDataDir,'QMatrix.npz'), sparse.csr_matrix(QMatrix))
        df.to_csv(os.path.join(self.LCDataDir,'studentSubmitRecords.csv'), sep='\t', index=False)
        saveDict(StaticInformation,self.LCDataDir,'StaticInformation.json')
        saveDict(userInformation,self.LCDataDir,'userInformation.json')
        saveDict(itemInformation,self.LCDataDir,'itemInformation.json')
        saveDict(knowledgeInformation,self.LCDataDir,'knowledgeInformation.json')
        saveDict(self.LC_params,self.LCDataDir,'parameters.json')

        return df, sparse.csr_matrix(QMatrix), StaticInformation, DictList


'''
#用户条件限制[最少做题数，最多做题数，最小通过率，最大通过率]
userLC = [10,500,0.1,1]
problemLC = [10,500,0,1]
#hdu原始数据里的最值，可以注释，不要删
low_time = "2018-06-01 00:00:00" 
high_time = "2018-11-29 00:00:00"
timeLC = [low_time, high_time]
a = _OJDataProcessor(userLC, problemLC, timeLC, True)
start = time.time()
[df, QMatrix, StaticInformation, DictList] = a.loadLCData()
end = time.time()
print("cost time: ", end - start)
print('**************QMatrix**************')
print(QMatrix.shape)
print('**************StaticInformation**************')
printDict(StaticInformation)
'''
