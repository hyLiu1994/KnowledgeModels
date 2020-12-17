import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from utils.this_queue import OurQueue
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from public import *
from KDDCupDataProcessor import _KDDCupDataProcessor
from OJDataProcessor import _OJDataProcessor
from AssistDataProcessor import _AssistDataProcessor

class _DataProcessor:
	# minTimestamp必须传值，默认值不管用
	def __init__(self, userLC, problemLC, timeLC, dataType = 'kdd', TmpDir = './data/'):
		if dataType == 'kdd':
			self.dataprocessor= _KDDCupDataProcessor(userLC, problemLC, timeLC, TmpDir = TmpDir)
		elif dataType == 'oj':
			self.dataprocessor= _OJDataProcessor(userLC, problemLC, timeLC, TmpDir = TmpDir)
		elif dataType == 'assist':
			self.dataprocessor= _AssistDataProcessor(userLC, problemLC, timeLC, TmpDir = TmpDir)
		elif dataType == 'math':
			self.dataprocessor= _MathDataProcessor(TmpDir = TmpDir)

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
			dict_data[str(run_id)]={'train':users_train,'test':users_test}

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

	def loadSparseDF(self, active_features = [], window_lengths = [3600 * 1e19, 3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600], verbose=True):
		"""Build sparse features dataset from dense dataset and q-matrix.

		Arguments:
		df -- dense dataset, output from one function from prepare_data.py (pandas DataFrame)
		Q_mat -- q-matrix, output from one function from prepare_data.py (sparse array)
		active_features -- features used to build the dataset (list of strings)
		tw -- useful when script is *not* called from command line.
		verbose -- if True, print information on the encoding process (bool)

		Output:
		sparse_df -- sparse dataset. The 4 first columns of sparse_df are just the same columns as in df.
		"""

		features_suffix = getFeaturesSuffix(active_features)
		SaveDir = os.path.join(self.LCDataDir, 'SparseFeatures')
		print(features_suffix)
		if os.path.exists(os.path.join(SaveDir, 'X-{:s}.npz'.format(features_suffix))):
			print ("存在现有的SparseFeatures, 直接读取")
			sparse_df = sparse.csr_matrix(sparse.load_npz(os.path.join(SaveDir, 'X-{:s}.npz'.format(features_suffix))))
			Length = loadDict(SaveDir,'Length-{:s}.json'.format(features_suffix))
			return sparse_df, Length
		print ("不存在现有的SparseFeatures, 重新生成")
		prepareFolder(SaveDir)

		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		QMatrix = QMatrix.toarray()

		NB_OF_TIME_WINDOWS = len(window_lengths)

		# Transform q-matrix into dictionary
		dict_q_mat = {i:set() for i in range(QMatrix.shape[0])}
		for elt in np.argwhere(QMatrix == 1):
			dict_q_mat[elt[0]].add(elt[1])

		X = {}
		Length = {}
		X['df'] = np.empty((0,4)) # Keep track of the original dataset

		X["skills"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
		Length["skills"] = QMatrix.shape[1]

		if 'lasttime_0kcsingle' in active_features:
			X['lasttime_0kcsingle'] = sparse.csr_matrix(np.empty((0, 1)))
			Length["lasttime_0kcsingle"] = 1

		X["lasttime_1kc"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
		Length["lasttime_1kc"] = QMatrix.shape[1]


		X['lasttime_2items'] = sparse.csr_matrix(np.empty((0, 1)))
		Length["lasttime_2items"] = 1

		X['lasttime_3sequence'] = sparse.csr_matrix(np.empty((0, 1)))
		Length["lasttime_3sequence"] = 1

		X['wins_1kc'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))
		Length["wins_1kc"] = QMatrix.shape[1] * NB_OF_TIME_WINDOWS
		
		X['wins_2items'] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
		Length["wins_2items"] = NB_OF_TIME_WINDOWS

		X['wins_3das3h'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
		Length["wins_3das3h"] = QMatrix.shape[1]

		X['wins_4das3hkc'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))
		Length["wins_4das3hkc"] = QMatrix.shape[1] * NB_OF_TIME_WINDOWS

		X['wins_5das3hitems'] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
		Length["wins_5das3hitems"] = NB_OF_TIME_WINDOWS

		X['attempts_1kc'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))
		Length["attempts_1kc"] = QMatrix.shape[1] * NB_OF_TIME_WINDOWS
		
		X['attempts_2items'] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
		Length["attempts_2items"] = NB_OF_TIME_WINDOWS

		X['attempts_3das3h'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
		Length["attempts_3das3h"] = QMatrix.shape[1]

		X['attempts_4das3hkc'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))
		Length["attempts_4das3hkc"] = QMatrix.shape[1] * NB_OF_TIME_WINDOWS

		X['attempts_5das3hitems'] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
		Length["attempts_5das3hitems"] = NB_OF_TIME_WINDOWS

		if 'failsdas3h' in active_features:
			X["failsdas3h"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
			Length["failsdas3h"] = QMatrix.shape[1]
 
		q_kc = defaultdict(lambda: OurQueue(window_lengths))  # Prepare counters for time windows kc_related
		q_item = defaultdict(lambda: OurQueue(window_lengths))  # item_related

		i = 0
		userNum = len(df['user_id'].unique())
		for stud_id in df['user_id'].unique():
			df_stud = df[df['user_id']==stud_id][['user_id', 'item_id', 'timestamp', 'correct']].copy()
			df_stud.sort_values(by='timestamp', inplace=True) # Sort values 
			df_stud = np.array(df_stud)

			X['df'] = np.vstack((X['df'], df_stud)) #rawdata 0-4列

			#skills
			if 'skills' in active_features:
				skills = QMatrix[df_stud[:,1].astype(int)].copy()
				X['skills'] = sparse.vstack([X['skills'],sparse.csr_matrix(skills)])

			kc_related = ['lasttime_0kcsingle', 'lasttime_1kc', 'wins_1kc', 'wins_4das3hkc', 'attempts_1kc', 'attempts_4das3hkc']
			item_related = ['lasttime_2items', 'wins_2items', 'wins_5das3hitems', 'attempts_2items']
			other = ['lasttime_3sequence', 'wins_3das3h', 'failsdas3h', 'attempts_3das3h', 'attempts_5das3hitems']

			skills = QMatrix[df_stud[:,1].astype(int)].copy()
			lasttime_0kcsingle = np.zeros((df_stud.shape[0], 1))
			lasttime_1kc = np.zeros((df_stud.shape[0], QMatrix.shape[1]))
			lasttime_2items = np.zeros((df_stud.shape[0], 1))
			lasttime_3sequence = np.zeros((df_stud.shape[0], 1))
			
			wins_1kc = np.zeros((df_stud.shape[0], QMatrix.shape[1] * NB_OF_TIME_WINDOWS))
			wins_2items = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
			wins_4das3hkc = np.zeros((df_stud.shape[0], QMatrix.shape[1] * NB_OF_TIME_WINDOWS))
			wins_5das3hitems = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))

			attempts_1kc = np.zeros((df_stud.shape[0], QMatrix.shape[1] * NB_OF_TIME_WINDOWS))
			attempts_2items = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
			attempts_4das3hkc = np.zeros((df_stud.shape[0], QMatrix.shape[1] * NB_OF_TIME_WINDOWS))
			attempts_5das3hitems = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
				
			for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
				if l != 0:
					lasttime_1kc[l, :] = lasttime_1kc[l-1, :]
					attempts_1kc[l, :] = attempts_1kc[l-1, :]
					lasttime_3sequence[l] = lasttime_3sequence[l-1]
				for skill_id in dict_q_mat[item_id]:
					if 'lasttime_0kcsingle' in active_features:
						lasttime_0kcsingle[l] = q_kc[stud_id, skill_id].get_last()
					lasttime_1kc[l, skill_id] = q_kc[stud_id, skill_id].get_last()
					lasttime_2items[l] = q_item[stud_id, item_id].get_counters(t)

					wins_1kc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = q_kc[stud_id, skill_id, "correct"].get_counters(t)
					wins_2items = q_item[stud_id, item_id, "correct"].get_counters(t)
					wins_4das3hkc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.log(1 + np.array(q_kc[stud_id, skill_id, "correct"].get_counters(t)))
					wins_5das3hitems[l] = np.log(1 + np.array(q_item[stud_id, item_id, "correct"].get_counters(t)))

					attempts_1kc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = q_kc[stud_id, skill_id].get_counters(t)
					attempts_2items[l] = q_item[stud_id, item_id].get_counters(t)
					attempts_4das3hkc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.log(1 + np.array(q_kc[stud_id, skill_id].get_counters(t)))
					attempts_5das3hitems[l] = np.log(1 + np.array(q_item[stud_id, item_id].get_counters(t)))
					
					q_kc[stud_id, skill_id].push(t)
					q_item[stud_id, item_id].push(t)
					if correct:
						q_kc[stud_id, item_id, "correct"].push(t)
						q_item[stud_id, item_id, "correct"].push(t)

			skills_temp = QMatrix[df_stud[:,1].astype(int)].copy()
			attempts_3das3h = np.multiply(np.cumsum(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp)),0)[:-1],skills_temp)
			wins_3das3h = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp)),
				np.hstack((np.array([0]),df_stud[:,3])).reshape(-1,1)),0)[:-1],skills_temp)
			
			if 'lasttime_0kcsingle' in active_features:
				X['lasttime_0kcsingle'] = sparse.vstack([X['lasttime_0kcsingle'],sparse.csr_matrix(lasttime_0kcsingle)])
			X['lasttime_1kc'] = sparse.vstack([X['lasttime_1kc'],sparse.csr_matrix(lasttime_1kc)])
			X['lasttime_2items'] = sparse.vstack([X['lasttime_2items'],sparse.csr_matrix(lasttime_2items)])
			X['wins_1kc'] = sparse.vstack([X['attempts_1kc'],sparse.csr_matrix(attempts_1kc)])
			X['wins_2items'] = sparse.vstack([X['attempts_1kc'],sparse.csr_matrix(attempts_1kc)])
			X['wins_3das3h'] = sparse.vstack([X['wins_3das3h'],sparse.csr_matrix(wins_3das3h)])
			X['wins_4das3hkc'] = sparse.vstack([X['attempts_1kc'],sparse.csr_matrix(attempts_1kc)])
			X['wins_5das3hitems'] = sparse.vstack([X['attempts_1kc'],sparse.csr_matrix(attempts_1kc)])
			X['attempts_1kc'] = sparse.vstack([X['attempts_1kc'],sparse.csr_matrix(attempts_1kc)])
			X['attempts_2items'] = sparse.vstack([X['attempts_2items'],sparse.csr_matrix(attempts_2items)])
			X['attempts_3das3h'] = sparse.vstack([X['attempts_3das3h'],sparse.csr_matrix(attempts_3das3h)])
			X['attempts_4das3hkc'] = sparse.vstack([X['attempts_4das3hkc'],sparse.csr_matrix(attempts_4das3hkc)])
			X['attempts_5das3hitems'] = sparse.vstack([X['attempts_5das3hitems'],sparse.csr_matrix(attempts_5das3hitems)])

			if "failsdas3h" in active_features:
				failsdas3h = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills.shape[1]),skills)),
					np.hstack((np.array([0]),1-df_stud[:,3])).reshape(-1,1)),0)[:-1],skills)
				X["failsdas3h"] = sparse.vstack([X["failsdas3h"],sparse.csr_matrix(failsdas3h)])
			i+=1
			if i%100 == 0:
				print(i,userNum)

		onehot = OneHotEncoder()
		if 'users' in active_features:
			X['users'] = onehot.fit_transform(X["df"][:,0].reshape(-1,1))
			if verbose:
				print("Users encoded.")
		if 'items' in active_features:
			X['items'] = onehot.fit_transform(X["df"][:,1].reshape(-1,1))
			if verbose:
				print("Items encoded.")

		'''for agent in active_features:
			print(agent)
			print(sparse.csr_matrix(X['df']).shape)
			print(X[agent].shape)
			sparse.hstack([sparse.csr_matrix(X['df']),X[agent]]).tocsr()
		'''


		sparse_df = sparse.hstack([sparse.csr_matrix(X['df']),sparse.hstack([X[agent] for agent in active_features])]).tocsr()
		# 此处时间窗口数无法保存
		sparse.save_npz(os.path.join(SaveDir, 'X-{:s}.npz'.format(features_suffix)), sparse_df)
		saveDict(Length, SaveDir, 'Length-{:s}.json'.format(features_suffix))
		return sparse_df, Length



userLC = [10,500,0.1,1]
problemLC = [10,500,0,1]
#hdu原始数据里的最值，可以注释，不要删
low_time = "2018-06-01 00:00:00" 
high_time = "2018-11-29 00:00:00"
timeLC = [low_time, high_time]
a = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir = "../data")
 
Features = {}
Features['users'] = False
Features['items'] = False
Features['skills'] = False
Features['lasttime_0kcsingle'] = False
Features['lasttime_1kc'] = True
Features['lasttime_2items'] = False
Features['lasttime_3sequence'] = False
Features['wins_1kc'] = True
Features['wins_2items'] = True
Features['wins_3das3h'] = True #用于das3h中特征
Features['wins_4das3hkc'] = True #用于das3h中特征
Features['wins_5das3hitems'] = True #用于das3h中特征
Features['failsdas3h'] = True
Features['attempts_1kc'] = True
Features['attempts_2items'] = True
Features['attempts_3das3h'] = True #用于das3h中特征
Features['attempts_4das3hkc'] = True #用于das3h中特征
Features['attempts_5das3hitems'] = True #用于das3h中特征

window_lengths = [3600]
#window_lengths = [3600 * 1e19, 3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]

active_features = [key for key, value in Features.items() if value]
# active_features = [key for key, value in Features.items()]
sparse_df, Length = a.loadSparseDF(active_features, window_lengths)
print('**************sparse_df**************')
print(sparse_df.shape)
printDict(Length)
print('**************statics**************')
printDict(a.dataprocessor.LC_params)




# assistments12
'''
userLC = [10,20]
problemLC = [10,20]
#assistments12原始数据里的最值，可以注释，不要删
low_time = "2012-09-01 00:00:00"
high_time = "2013-09-01 00:00:00"
timeLC = [low_time, high_time]

a = _DataProcessor(userLC, problemLC, timeLC, 'assit', TmpDir = "../data")
print('**************LC_params**************')
printDict(a.LC_params)
'''