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

class _MathDataProcessor(object):
	# 该数据集没有时间
	def __init__(self, datasetName = 'math1',  TmpDir = "./data/"):
		self.datasetName = datasetName

		self.LC_params={}

		self.TmpDir =  TmpDir
		self.RawDataDir = os.path.join(self.TmpDir, 'rawData', self.datasetName)
		self.RawDataName = 'data.txt'
		self.RawKnowledge2Problem = 'q.txt'
		self.RawKCName = 'qnames.txt'
		self.LCDataDir = os.path.join(self.TmpDir, 'LCData',self.datasetName+getLegend(self.LC_params))

	def loadLCData(self):
		# 所有的字典都是id2str

		flag = 0
		if not os.path.exists(os.path.join(self.LCDataDir, 'studentSubmitRecords.csv')):
			flag = 1
		if not os.path.exists(os.path.join(self.LCDataDir, 'QMatrix.npz')):
			flag = 1
		if not os.path.exists(os.path.join(self.LCDataDir, 'StaticInformation.json')):
			flag = 1
		if not os.path.exists(os.path.join(self.LCDataDir, 'knowledgeInformation.json')):
			flag = 1

		if flag == 0:
			df = pd.read_csv(os.path.join(self.LCDataDir, 'studentSubmitRecords.csv'), sep='\t')
			QMatrix = sparse.load_npz(os.path.join(self.LCDataDir,'QMatrix.npz'))
			StaticInformation = loadDict(self.LCDataDir,'StaticInformation.json')
			knowledgeInformation = loadDict(self.LCDataDir,'knowledgeInformation.json')
			DictList = [knowledgeInformation]
			return df, QMatrix, StaticInformation, DictList

		prepareFolder(self.LCDataDir)

		df = {'user_id':[], 'item_id':[], 'timestamp':[], 'correct':[]}

		with open(os.path.join(self.RawDataDir, self.RawDataName), 'r', encoding = 'UTF-8') as f:
			indexU = 0
			for line in f: 
				lineArray = line.split('\t')
				indexI = 0
				for item in lineArray:
					df['user_id'].append(indexU)
					df['item_id'].append(indexI)
					df['timestamp'].append(0)
					df['correct'].append(item)
					indexI += 1
				indexU += 1
		df = pd.DataFrame(df)

		# Build Q-matrix
		knowledgeInformation = {}
		df_kc = pd.read_csv(os.path.join(self.RawDataDir, self.RawKCName), delimiter='\t', engine='python')
		print(df_kc)
		for item in zip(df_kc[df_kc.columns[0]],df_kc[df_kc.columns[1]]):
			knowledgeInformation[item[0]] = item[1] 
		knowledgeId = 0
		QMatrix = np.zeros((len(df['item_id'].unique()), len(df_kc)))
		numKCs = 0
		with open(os.path.join(self.RawDataDir, self.RawKnowledge2Problem), 'r', encoding = 'UTF-8') as f:
			indexI = 0
			for line in f: 
				lineArray = line.split('\t')
				indexK = 0
				for item in lineArray:
					QMatrix[indexI][indexK] = item
					indexK += 1
					if int(item) == 1:
						numKCs += 1
				indexI += 1

		StaticInformation = {}
		StaticInformation['userNum'] = len(df['user_id'].unique())
		StaticInformation['itemNum'] = len(df['item_id'].unique())
		StaticInformation['knowledgeNum'] = QMatrix.shape[1]
		StaticInformation['recordsNum'] = df.shape[0]

		StaticInformation['aveUserSubmit'] = df.shape[0] / len(df['user_id'].unique())
		StaticInformation['aveitemNumSubmit'] = df.shape[0] / len(df['item_id'].unique())
		StaticInformation['aveItemContainKnowledge'] = numKCs / len(df['item_id'].unique())

		DictList = [knowledgeInformation]

		# Save data
		sparse.save_npz(os.path.join(self.LCDataDir,'QMatrix.npz'), sparse.csr_matrix(QMatrix))
		df.to_csv(os.path.join(self.LCDataDir,'studentSubmitRecords.csv'), sep='\t', index=False)
		saveDict(StaticInformation,self.LCDataDir,'StaticInformation.json')
		saveDict(knowledgeInformation,self.LCDataDir,'knowledgeInformation.json')
		saveDict(self.LC_params,self.LCDataDir,'parameters.json')

		return df, QMatrix, StaticInformation, DictList



a = _MathDataProcessor(TmpDir = "../data")
start = time.time()
[df, QMatrix, StaticInformation, DictList] = a.loadLCData()
end = time.time()
print("cost time: ", end - start)
print('**************QMatrix**************')
print(QMatrix.shape)
print('**************StaticInformation**************')
printDict(StaticInformation)


