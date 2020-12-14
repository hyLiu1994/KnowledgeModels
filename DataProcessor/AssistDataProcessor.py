import os
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import KFold
from public import *

class _AssistDataProcessor:
    def __init__(self, userLC, problemLC, timeLC, drop_duplicates = True, remove_nan_skills = True, datasetName = 'assistments12', TmpDir = "./data/"):

        self.datasetName = datasetName

        self.LC_params={}
        self.LC_params['userLC'] = userLC
        self.LC_params['problemLC'] = problemLC
        self.LC_params['timeLC'] = transferStrT2Dir(timeLC)
        self.LC_params['dropDup'] = drop_duplicates
        self.LC_params['remoNanSkill'] = remove_nan_skills

        self.TmpDir = TmpDir
        self.RawDataDir = os.path.join(self.TmpDir, 'rawData', self.datasetName)
        self.RawDataName = 'data.csv'
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
        if not os.path.exists(os.path.join(self.LCDataDir, 'knowledgeInformation.json')):
            flag = 1

        if flag == 0:
            df = pd.read_csv(os.path.join(self.LCDataDir, 'studentSubmitRecords.csv'), sep='\t')
            QMatrix = sparse.load_npz(os.path.join(self.LCDataDir,'QMatrix.npz'))
            StaticInformation = loadDict(self.LCDataDir,'StaticInformation.json')
            userInformation = loadDict(self.LCDataDir,'userInformation.json')
            knowledgeInformation = loadDict(self.LCDataDir,'knowledgeInformation.json')
            DictList = [userInformation, knowledgeInformation]
            return df, QMatrix, StaticInformation, DictList

        prepareFolder(self.LCDataDir)

        df = pd.read_csv(os.path.join(self.RawDataDir, self.RawDataName))

        print(df.columns)

        # 1 timeLC
        df["timestamp"] = df["start_time"]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(df['timestamp'].min(), df['timestamp'].max())
        df['timestamp'] = df['timestamp'] - self.minTimestamp
        df['timestamp'] = df['timestamp'].apply(lambda x: x.total_seconds()).astype(np.int64)
        df = df[df.timestamp >= self.timeLC[0]]
        df = df[df.timestamp <= self.timeLC[1]]

        # 2 userLC
        df = df.groupby('user_id').filter(lambda x: len(x) >= self.LC_params['userLC'][0])
        df = df.groupby('user_id').filter(lambda x: len(x) <= self.LC_params['userLC'][1])

        # 3 problemLC
        df = df.groupby('problem_id').filter(lambda x: len(x) >= self.LC_params['problemLC'][0])
        df = df.groupby('problem_id').filter(lambda x: len(x) <= self.LC_params['problemLC'][1])

        df.sort_values(by = 'timestamp', inplace = True)

        # 这里的去重指的是一个人同时提交同一道题
        if self.LC_params['dropDup']:
            df.drop_duplicates(subset=['user_id', 'problem_id', 'timestamp'], inplace=True)
        
        if self.LC_params['remoNanSkill']:
            df = df[~df['skill_id'].isnull()]
        else:
            df.ix[df['skill_id'].isnull(), 'skill_id'] = 'NaN'

        userInformation = createDictBydf(df, 'user_id')
        itemInformation = createDictBydf(df, 'problem_id')
        knowledgeInformation = createDictBydf(df, 'skill_id')
        user2id = reverseDict(userInformation)
        item2id = reverseDict(itemInformation)
        skill2id = reverseDict(knowledgeInformation)

        # Transform ids into numeric
        df['user_id'] = df['user_id'].apply(lambda x: user2id[x])
        df['item_id'] = df['problem_id'].apply(lambda x: item2id[x])
        df['skill_id'] = df['skill_id'].apply(lambda x: skill2id[x])

        # # Build Q-matrix
        QMatrix = np.zeros((len(df["item_id"].unique()), len(df["skill_id"].unique())))
        item_skill = np.array(df[["item_id", "skill_id"]])
        for i in range(len(item_skill)):
            QMatrix[item_skill[i,0],item_skill[i,1]] = 1
        numKCs = str(QMatrix.tolist()).count("1")

        StaticInformation = {}
        StaticInformation['userNum'] = len(df['user_id'].unique())
        StaticInformation['itemNum'] = len(df['item_id'].unique())
        StaticInformation['knowledgeNum'] = len(df['skill_id'].unique())
        StaticInformation['recordsNum'] = df.shape[0]

        StaticInformation['aveUserSubmit'] = df.shape[0] / len(df['user_id'].unique())
        StaticInformation['aveitemNumSubmit'] = df.shape[0] / len(df['item_id'].unique())
        StaticInformation['aveItemContainKnowledge'] = numKCs / len(df['item_id'].unique())

        DictList = [userInformation, knowledgeInformation]

        df = df[['user_id', 'item_id', 'timestamp', 'correct']]
        df = df[df.correct.isin([0,1])] # Remove potential continuous outcomes
        df['correct'] = df['correct'].astype(np.int32) # Cast outcome as int32

        # Save data
        sparse.save_npz(os.path.join(self.LCDataDir,'QMatrix.npz'), sparse.csr_matrix(QMatrix))
        df.to_csv(os.path.join(self.LCDataDir,'studentSubmitRecords.csv'), sep='\t', index=False)
        saveDict(StaticInformation,self.LCDataDir,'StaticInformation.json')
        saveDict(userInformation,self.LCDataDir,'userInformation.json')
        saveDict(knowledgeInformation,self.LCDataDir,'knowledgeInformation.json')
        saveDict(self.LC_params,self.LCDataDir,'parameters.json')

        return df, QMatrix, StaticInformation, DictList



'''
userLC = [10,20]
problemLC = [10,20]
#assistments12原始数据里的最值，可以注释，不要删
low_time = "2012-09-01 00:00:00"
high_time = "2013-09-01 00:00:00"
timeLC = [low_time, high_time]
a = _AssistDataProcessor(userLC, problemLC, timeLC, TmpDir = "../data")
print('**************LC_params**************')
printDict(a.LC_params)
[df, QMatrix, StaticInformation, DictList] = a.loadLCData()
print('**************StaticInformation**************')
printDict(StaticInformation)
'''
