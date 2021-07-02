import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
from utils.this_queue import OurQueue, OurQueueDas3h
from collections import defaultdict, Counter
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from public import *
from KDDCupDataProcessor import _KDDCupDataProcessor
from OJDataProcessor import _OJDataProcessor
from AssistDataProcessor import _AssistDataProcessor

class _DataProcessor:
	# minTimestamp必须传值，默认值不管用
	def __init__(self, userLC, problemLC, timeLC, dataType = 'kdd', TmpDir = './data/', datasetName = 'algebra08'):
		if dataType == 'kdd':
			self.dataprocessor= _KDDCupDataProcessor(userLC, problemLC, timeLC, datasetName = datasetName, TmpDir = TmpDir)
		elif dataType == 'oj':
			self.dataprocessor= _OJDataProcessor(userLC, problemLC, timeLC, TmpDir = TmpDir)
		elif dataType == 'assist':
			self.dataprocessor= _AssistDataProcessor(userLC, problemLC, timeLC, TmpDir = TmpDir)

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

	def getLengthDistribution(self, kFold):
		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		dictionary={'stud_id':[],'length':[]}
		print(df.columns)
		for stud_id in df['user_id'].unique():
			df_stud = df[df['user_id']==stud_id][['user_id', 'item_id', 'timestamp', 'correct']].copy()
			print(stud_id, len(df_stud))
			dictionary['stud_id'].append(stud_id)
			dictionary['length'].append(len(df_stud))
		df_length = pd.DataFrame(dictionary)
		# df_length.to_csv(os.path.join(self.LCDataDir,'studentSubmitLength.csv'), sep='\t', index=False)

		dict_data = self.loadSplitInfo(kFold)

		return df_length, dict_data

	def loadAKTData_5F(self, dataset_params, fold_id):
		trainRate = dataset_params["trainRate"]
		batch_size = dataset_params["batch_size"]
		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		item_num = StaticInformation['itemNum']
		dict_data = self.loadSplitInfo(dataset_params['kFold'])
		train_uid = dict_data[str(fold_id)]['train']
		test_uid = dict_data[str(fold_id)]['test']
		train_df = df[df['user_id'].isin(train_uid)]
		test_df = df[df['user_id'].isin(test_uid)]
		train_dataset = self.df2dataset_AKT(train_df, StaticInformation, batch_size)
		test_dataset = self.df2dataset_AKT(test_df, StaticInformation, batch_size)
		return train_dataset, test_dataset, QMatrix
		

	def df2dataset_AKT(self, df, StaticInformation, batch_size):
		df['item_id'] = df['item_id'] + 1
		data = df.groupby('user_id').apply(lambda r: (r['item_id'].values, r['timestamp'].values, r['correct'].values))
		problem_num = len(df['item_id'].unique())

		item, timestamp, correct = list(), list(), list()
		def split_seq(inputs, max_len=500):
			seq_len = len(inputs)
			subseq_num = seq_len // max_len + int(seq_len % max_len != 0)
			subseq = list()
			for i in range(subseq_num):
				subseq.append(inputs[i*max_len:(i+1)*max_len])
			return subseq

		for it in data:
			item += split_seq(it[0])
			timestamp += split_seq(it[1] / (6 * 30 * 24 * 3600))
			correct += split_seq(it[2])
		
		def __to_dataset(data):
			data = tf.ragged.constant(data)
			data = tf.data.Dataset.from_tensor_slices(data)
			data = data.map(lambda x: x)
			return data

		item = __to_dataset(item)
		timestamp = __to_dataset(timestamp)
		correct = __to_dataset(correct)

		dataset = tf.data.Dataset.zip((item, timestamp, correct))
        # dtype
		dataset = dataset.map(lambda item, timestamp, correct: (tf.cast(item, dtype=tf.float32),tf.cast(timestamp, dtype=tf.float32), tf.cast(correct, dtype=tf.float32)))
		# dim
		dataset = dataset.map(lambda item, timestamp, correct: (tf.expand_dims(item, axis=-1),tf.expand_dims(timestamp, axis=-1), tf.expand_dims(correct, axis=-1), correct))
		# concat
		dataset = dataset.map(lambda item, timestamp, correct, label: (tf.concat([item, timestamp, correct], axis=-1), label))

		dataset = dataset.padded_batch(batch_size,  drop_remainder=False)
		return dataset

	def loadAKTData(self, dataset_params):
		trainRate = dataset_params["trainRate"]
		batch_size = dataset_params["batch_size"]
		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		data = df.groupby('user_id').apply(lambda r: (r['item_id'].values, r['timestamp'].values, r['correct'].values))
		df['item_id'] = df['item_id'] + 1
		problem_num = len(df['item_id'].unique())

		item, timestamp, correct = list(), list(), list()


		def split_seq(inputs, max_len=500):
			seq_len = len(inputs)
			subseq_num = seq_len // max_len + int(seq_len % max_len != 0)
			subseq = list()
			for i in range(subseq_num):
				subseq.append(inputs[i*max_len:(i+1)*max_len])
			return subseq

		for it in data:
			item += split_seq(it[0])
			timestamp += split_seq(it[1] / (24 * 3600))
			correct += split_seq(it[2])
		
		def __to_dataset(data):
			data = tf.ragged.constant(data)
			data = tf.data.Dataset.from_tensor_slices(data)
			data = data.map(lambda x: x)
			return data

		item = __to_dataset(item)
		timestamp = __to_dataset(timestamp)
		correct = __to_dataset(correct)

		dataset = tf.data.Dataset.zip((item, timestamp, correct))
                # dtype
		dataset = dataset.map(lambda item, timestamp, correct: (tf.cast(item, dtype=tf.float32),tf.cast(timestamp, dtype=tf.float32), tf.cast(correct, dtype=tf.float32)))
		# dim
		dataset = dataset.map(lambda item, timestamp, correct: (tf.expand_dims(item, axis=-1),tf.expand_dims(timestamp, axis=-1), tf.expand_dims(correct, axis=-1), correct))
		# concat
		dataset = dataset.map(lambda item, timestamp, correct, label: (tf.concat([item, timestamp, correct], axis=-1), label))

		def __split_dataset(dataset, trainRate):
			if tf.__version__ == '2.2.0':
				data_size = tf.data.experimental.cardinality(dataset).numpy()
			else:
				data_size = dataset.cardinality().numpy()
			train_size = int(data_size * trainRate)
			print("train_size: ", train_size)
			print("test_size: ", data_size -  train_size)

			train_dataset = dataset.take(train_size)
			test_dataset = dataset.skip(train_size)
			return train_dataset, test_dataset

		train_dataset, test_dataset = __split_dataset(dataset, trainRate)
		train_dataset = train_dataset.padded_batch(batch_size,  drop_remainder=False)
		test_dataset = test_dataset.padded_batch(batch_size,  drop_remainder=False)
		return train_dataset, test_dataset, QMatrix

	def df2dataset_DKT(self, df, StaticInformation, batch_size):
		df = df.drop(['timestamp'],axis=1)
		df['item_id'] = df['item_id'] + 1
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
		dataset = dataset.map(lambda inputs, data, label: (tf.cast(inputs, dtype=tf.int32), tf.cast(data, dtype=tf.float32), tf.cast(label, dtype=tf.float32)))
		# dim
		dataset = dataset.map(lambda inputs, data, label: (inputs, tf.expand_dims(data, axis=-1), tf.expand_dims(label, axis=-1))
							  )
		# concat
		dataset = dataset.map(lambda inputs, data, label: (inputs, tf.concat([data, label], axis=-1))
							  )

		dataset = dataset.padded_batch(batch_size, drop_remainder=False)
		return dataset

	def df2dataset_DKVMN(self, df, StaticInformation, batch_size):
		df = df.drop(['timestamp'],axis=1)
		item_num = StaticInformation['itemNum']
		df['skill_id_with_correct'] = df['correct'] * item_num + df['item_id']
		df['item_id'] = df['item_id'] + 1
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

		dataset = dataset.padded_batch(batch_size, drop_remainder=False)
		return dataset

	def loadDKTbatchData_5F(self, dataset_params, fold_id):
		trainRate = dataset_params["trainRate"]
		batch_size = dataset_params["batch_size"]
		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		item_num = StaticInformation['itemNum']
		dict_data = self.loadSplitInfo(dataset_params['kFold'])
		train_uid = dict_data[str(fold_id)]['train'] 
		test_uid =  dict_data[str(fold_id)]['test']
		train_df = df[df['user_id'].isin(train_uid)]
		test_df = df[df['user_id'].isin(test_uid)]
		train_dataset = self.df2dataset_DKT(train_df, StaticInformation, batch_size)
		test_dataset = self.df2dataset_DKT(test_df, StaticInformation, batch_size)
		return train_dataset, test_dataset, item_num


	def loadDKTbatchData(self, dataset_params):
		trainRate = dataset_params["trainRate"]
		batch_size = dataset_params["batch_size"]
		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		df = df.drop(['timestamp'],axis=1)
		df['item_id'] = df['item_id'] + 1
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

	def loadDKVMNbatchData_5F(self, dataset_params, fold_id):
		trainRate = dataset_params["trainRate"]
		batch_size = dataset_params["batch_size"]
		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		item_num = StaticInformation['itemNum']
		dict_data = self.loadSplitInfo(dataset_params['kFold'])
		train_uid = dict_data[str(fold_id)]['train'] 
		test_uid =  dict_data[str(fold_id)]['test']
		train_df = df[df['user_id'].isin(train_uid)]
		test_df = df[df['user_id'].isin(test_uid)]
		train_dataset = self.df2dataset_DKVMN(train_df, StaticInformation, batch_size)
		test_dataset = self.df2dataset_DKVMN(test_df, StaticInformation, batch_size)
		return train_dataset, test_dataset, item_num

	def loadDAS3HData(self, active_features = ['skills'], features_suffix = '', trainRate = 0.8, tw = None, verbose = True):

		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		df.reset_index(inplace=True, drop=True) # Add unique identifier of the row
		df["inter_id"] = df.index
		SaveDir = os.path.join(self.LCDataDir, 'das3h')
		prepareFolder(SaveDir)

		if os.path.exists(os.path.join(SaveDir, 'X-{:s}.npz'.format(features_suffix))):
			print ("存在现有的SparseFeatures, 直接读取")
			sparse_df = sparse.csr_matrix(sparse.load_npz(os.path.join(SaveDir, 'X-{:s}.npz'.format(features_suffix))))

			all_users = df['user_id'].unique()
			users_train, users_test = train_test_split(all_users, train_size = trainRate, shuffle = False)

			dict_data = {'0':{}}
			dict_data['0']={'train':users_train,'test':users_test}

			saveDict(dict_data,self.LCDataDir, 'splitedInformation_trainRate('+str(int(trainRate*100))+').json')
			printDict(dict_data)
			print(dict_data)
			return sparse_df, dict_data

		# Transform q-matrix into dictionary
		dict_q_mat = {i:set() for i in range(QMatrix.shape[0])}
		for elt in np.argwhere(QMatrix == 1):
			dict_q_mat[elt[0]].add(elt[1])

		NB_OF_TIME_WINDOWS = 5

		X={}
		if 'skills' in active_features:
			X["skills"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
		if 'attempts' in active_features:
			if tw == "tw_kc":
				X["attempts"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1]*NB_OF_TIME_WINDOWS)))
			elif tw == "tw_items":
				X["attempts"] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
			else:
				X["attempts"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
		if 'wins' in active_features:
			if tw == "tw_kc":
				X["wins"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1]*NB_OF_TIME_WINDOWS)))
			elif tw == "tw_items":
				X["wins"] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
			else:
				X["wins"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
		if 'fails' in active_features:
			X["fails"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))

		X['df'] = np.empty((0,5)) # Keep track of the original dataset

		q = defaultdict(lambda: OurQueueDas3h())  # Prepare counters for time windows
		for stud_id in df["user_id"].unique():
			df_stud = df[df["user_id"]==stud_id][["user_id", "item_id", "timestamp", "correct","inter_id"]].copy()
			df_stud.sort_values(by="timestamp", inplace=True) # Sort values 
			df_stud = np.array(df_stud)
			X['df'] = np.vstack((X['df'], df_stud))

			if 'skills' in active_features:
				skills_temp = QMatrix[df_stud[:,1].astype(int)].copy()
				X['skills'] = sparse.vstack([X["skills"],sparse.csr_matrix(skills_temp)])
			if "attempts" in active_features:
				skills_temp = QMatrix[df_stud[:,1].astype(int)].copy()
				if tw == "tw_kc":
					attempts = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS*QMatrix.shape[1]))
					for l, (item_id, t) in enumerate(zip(df_stud[:,1], df_stud[:,2])):
						for skill_id in dict_q_mat[item_id]:
							attempts[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.log(1 + \
								np.array(q[stud_id, skill_id].get_counters(t)))
							q[stud_id, skill_id].push(t)
				elif tw == "tw_items":
					attempts = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
					for l, (item_id, t) in enumerate(zip(df_stud[:,1], df_stud[:,2])):
						attempts[l] = np.log(1 + np.array(q[stud_id, item_id].get_counters(t)))
						q[stud_id, item_id].push(t)
				else:
					attempts = np.multiply(np.cumsum(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp.toarray())),0)[:-1],skills_temp.toarray())
				X['attempts'] = sparse.vstack([X['attempts'],sparse.csr_matrix(attempts)])
			if "wins" in active_features:
				skills_temp = QMatrix[df_stud[:,1].astype(int)].copy()
				if tw == "tw_kc":
					wins = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS*QMatrix.shape[1]))
					for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
						for skill_id in dict_q_mat[item_id]:
							wins[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = np.log(1 + \
								np.array(q[stud_id, skill_id, "correct"].get_counters(t)))
							if correct:
								q[stud_id, skill_id, "correct"].push(t)
				elif tw == "tw_items":
					wins = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
					for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
						wins[l] = np.log(1 + np.array(q[stud_id, item_id, "correct"].get_counters(t)))
						if correct:
							q[stud_id, item_id, "correct"].push(t)
				else:
					'''
					print(np.zeros(skills_temp.shape[1]))
					print(skills_temp)
					print(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp.toarray())))
					'''
					wins = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp.toarray())),np.hstack((np.array([0]),df_stud[:,3])).reshape(-1,1)),0)[:-1],skills_temp.toarray())

				X['wins'] = sparse.vstack([X['wins'],sparse.csr_matrix(wins)])
			if "fails" in active_features:
				skills_temp = QMatrix[df_stud[:,1].astype(int)].copy()
				fails = np.multiply(np.cumsum(np.multiply(np.vstack((np.zeros(skills_temp.shape[1]),skills_temp.toarray())),np.hstack((np.array([0]),1-df_stud[:,3])).reshape(-1,1)),0)[:-1],skills_temp.toarray())
				X["fails"] = sparse.vstack([X["fails"],sparse.csr_matrix(fails)])
			if verbose:
				print(X["df"].shape)

		onehot = OneHotEncoder()
		if 'users' in active_features:
			X['users'] = onehot.fit_transform(X["df"][:,0].reshape(-1,1))
			if verbose:
				print("Users encoded.")
		if 'items' in active_features:
			X['items'] = onehot.fit_transform(X["df"][:,1].reshape(-1,1))
			if verbose:
				print("Items encoded.")
		sparse_df = sparse.hstack([sparse.csr_matrix(X['df']),sparse.hstack([X[agent] for agent in active_features])]).tocsr()

		all_users = df['user_id'].unique()
		users_train, users_test = train_test_split(all_users, train_size = trainRate, shuffle = False)

		dict_data = {'0':{}}
		dict_data['0']={'train':users_train,'test':users_test}

		sparse.save_npz(os.path.join(SaveDir, 'X-{:s}.npz'.format(features_suffix)), sparse_df)
		saveDict(dict_data,self.LCDataDir, 'splitedInformation_trainRate('+str(int(trainRate*100))+').json')
		return sparse_df, dict_data


	def loadSparseDF(self, active_features = ['skills'], window_lengths = [3600 * 1e19, 3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600], all_features = ['users', 'items', 'skills', 'lasttime_1kc', 'lasttime_2items', 'lasttime_3sequence', 'interval_1kc', 'interval_2items', 'interval_3sequence', 'wins_1kc', 'wins_2items', 'fails_1kc', 'fails_2items', 'attempts_1kc', 'attempts_2items']):

		"""Build sparse features dataset from dense dataset and q-matrix.

		Arguments:
		df -- dense dataset, output from one function from prepare_data.py (pandas DataFrame)
		Q_mat -- q-matrix, output from one function from prepare_data.py (sparse array)
		active_features -- features used to build the dataset (list of strings)
		tw -- useful when script is *not* called from command line.

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

		prepareFolder(SaveDir)

		flag = 1
		isFeatureExist = {}
		if not os.path.exists(os.path.join(SaveDir, 'X.npz')):
			flag = 0
		for agent in active_features:
			f = getFeaturesSuffix([agent])
			if not os.path.exists(os.path.join(SaveDir, 'X-{:s}.npz'.format(f))):
				isFeatureExist[agent] = 0
				flag = 0
			else:
				isFeatureExist[agent] = 1

		if flag == 0:
			print ("不存在所有的SparseFeatures, 重新生成")

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

			X["lasttime_1kc"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
			Length["lasttime_1kc"] = QMatrix.shape[1]


			X['lasttime_2items'] = sparse.csr_matrix(np.empty((0, 1)))
			Length["lasttime_2items"] = 1

			X['lasttime_3sequence'] = sparse.csr_matrix(np.empty((0, 1)))
			Length["lasttime_3sequence"] = 1

			X["interval_1kc"] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1])))
			Length["interval_1kc"] = QMatrix.shape[1]


			X['interval_2items'] = sparse.csr_matrix(np.empty((0, 1)))
			Length["interval_2items"] = 1

			X['interval_3sequence'] = sparse.csr_matrix(np.empty((0, 1)))
			Length["interval_3sequence"] = 1


			X['wins_1kc'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))
			Length["wins_1kc"] = QMatrix.shape[1] * NB_OF_TIME_WINDOWS
				
			X['wins_2items'] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
			Length["wins_2items"] = NB_OF_TIME_WINDOWS

			X['attempts_1kc'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))
			Length["attempts_1kc"] = QMatrix.shape[1] * NB_OF_TIME_WINDOWS
				
			X['attempts_2items'] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
			Length["attempts_2items"] = NB_OF_TIME_WINDOWS


			X['fails_1kc'] = sparse.csr_matrix(np.empty((0, QMatrix.shape[1] * NB_OF_TIME_WINDOWS)))
			Length["fails_1kc"] = QMatrix.shape[1] * NB_OF_TIME_WINDOWS
				
			X['fails_2items'] = sparse.csr_matrix(np.empty((0, NB_OF_TIME_WINDOWS)))
			Length["fails_2items"] = NB_OF_TIME_WINDOWS

			q_kc = defaultdict(lambda: OurQueue(window_lengths))  # Prepare counters for time windows kc_related
			q_item = defaultdict(lambda: OurQueue(window_lengths))  # item_related

			i = 0
			userNum = len(df['user_id'].unique())
			print("userNum:", userNum)

			for stud_id in df['user_id'].unique():
				df_stud = df[df['user_id']==stud_id][['user_id', 'item_id', 'timestamp', 'correct']].copy()
				df_stud.sort_values(by='timestamp', inplace=True) # Sort values 
				df_stud = np.array(df_stud)

				X['df'] = np.vstack((X['df'], df_stud)) #rawdata 0-4列

				skills = QMatrix[df_stud[:,1].astype(int)].copy()
				X['skills'] = sparse.vstack([X['skills'],sparse.csr_matrix(skills)])

				lasttime_1kc = np.zeros((df_stud.shape[0], QMatrix.shape[1]))
				lasttime_2items = np.zeros((df_stud.shape[0], 1))
				lasttime_3sequence = np.zeros((df_stud.shape[0], 1))

				interval_1kc = np.zeros((df_stud.shape[0], QMatrix.shape[1]))
				interval_2items = np.zeros((df_stud.shape[0], 1))
				interval_3sequence = np.zeros((df_stud.shape[0], 1))

				
				wins_1kc = np.zeros((df_stud.shape[0], QMatrix.shape[1] * NB_OF_TIME_WINDOWS))
				wins_2items = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))

				fails_1kc = np.zeros((df_stud.shape[0], QMatrix.shape[1] * NB_OF_TIME_WINDOWS))
				fails_2items = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))

				attempts_1kc = np.zeros((df_stud.shape[0], QMatrix.shape[1] * NB_OF_TIME_WINDOWS))
				attempts_2items = np.zeros((df_stud.shape[0], NB_OF_TIME_WINDOWS))
					
				for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
					if l != 0:
						lasttime_3sequence[l] = lasttime_3sequence[l-1]
						interval_3sequence[l] = t - lasttime_3sequence[l]
					for skill_id in range(QMatrix.shape[1]):
					#for skill_id in dict_q_mat[item_id]:
						lasttime_1kc[l, skill_id] = q_kc[stud_id, skill_id].get_last()
						lasttime_2items[l] = q_item[stud_id, item_id].get_last()

						interval_1kc[l, skill_id] = t - q_kc[stud_id, skill_id].get_last()
						interval_2items[l] = t - q_item[stud_id, item_id].get_last()

						wins_1kc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = q_kc[stud_id, skill_id, "correct"].get_counters(t)
						wins_2items[l] = q_item[stud_id, item_id, "correct"].get_counters(t)						

						attempts_1kc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = q_kc[stud_id, skill_id].get_counters(t)
						attempts_2items[l] = q_item[stud_id, item_id].get_counters(t)

						fails_1kc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] = attempts_1kc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS] - wins_1kc[l, skill_id*NB_OF_TIME_WINDOWS:(skill_id+1)*NB_OF_TIME_WINDOWS]
						fails_2items[l] = attempts_2items[l] - wins_2items[l]

						if skill_id in dict_q_mat[item_id]:
							q_kc[stud_id, skill_id].push(t)
							if correct:
								q_kc[stud_id, skill_id, "correct"].push(t)
					q_item[stud_id, item_id].push(t)
					if correct:
						q_item[stud_id, item_id, "correct"].push(t)


				X['lasttime_1kc'] = sparse.vstack([X['lasttime_1kc'],sparse.csr_matrix(lasttime_1kc)])
				X['lasttime_2items'] = sparse.vstack([X['lasttime_2items'],sparse.csr_matrix(lasttime_2items)])
				X['lasttime_3sequence'] = sparse.vstack([X['lasttime_3sequence'],sparse.csr_matrix(lasttime_3sequence)])

				X['interval_1kc'] = sparse.vstack([X['interval_1kc'],sparse.csr_matrix(interval_1kc)])
				X['interval_2items'] = sparse.vstack([X['interval_2items'],sparse.csr_matrix(interval_2items)])
				X['interval_3sequence'] = sparse.vstack([X['interval_3sequence'],sparse.csr_matrix(interval_3sequence)])

				X['wins_1kc'] = sparse.vstack([X['wins_1kc'],sparse.csr_matrix(wins_1kc)])
				X['wins_2items'] = sparse.vstack([X['wins_2items'],sparse.csr_matrix(wins_2items)])

				X['attempts_1kc'] = sparse.vstack([X['attempts_1kc'],sparse.csr_matrix(attempts_1kc)])
				X['attempts_2items'] = sparse.vstack([X['attempts_2items'],sparse.csr_matrix(attempts_2items)])


				X['fails_1kc'] = sparse.vstack([X['fails_1kc'],sparse.csr_matrix(fails_1kc)])
				X['fails_2items'] = sparse.vstack([X['fails_2items'],sparse.csr_matrix(fails_2items)])
				i+=1
				if i%100 == 0:
					print(i,userNum)


			onehot = OneHotEncoder()
			X['users'] = onehot.fit_transform(X["df"][:,0].reshape(-1,1))
			Length['users'] = len(df['user_id'].unique())
			print("Users encoded.")
			X['items'] = onehot.fit_transform(X["df"][:,1].reshape(-1,1))
			Length['items'] = len(df['item_id'].unique())
			print("Items encoded.")


			print(all_features)
			for agent in all_features:
				f = getFeaturesSuffix([agent])
				if not os.path.exists(os.path.join(SaveDir, 'X-{:s}.npz'.format(f))):
					single = sparse.hstack([sparse.csr_matrix(X['df']),sparse.csr_matrix(X[agent])]).tocsr()
					sparse.save_npz(os.path.join(SaveDir, 'X-{:s}.npz'.format(f)), single)
					print(agent, ' saved.')
					saveDict({agent:Length[agent]}, SaveDir, 'Length-{:s}.json'.format(f))

			if not os.path.exists(os.path.join(SaveDir, 'X.npz')):
				single = sparse.csr_matrix(X['df']).tocsr()
				sparse.save_npz(os.path.join(SaveDir, 'X.npz'.format(f)), single)

			sparse_df = sparse.hstack([sparse.csr_matrix(X['df']),sparse.hstack([X[agent] for agent in active_features])]).tocsr()
			length = {}
			for key,value in Length.items():
				if key in active_features:
					length[key] = value
			# 此处时间窗口数无法保存
			sparse.save_npz(os.path.join(SaveDir, 'X-{:s}.npz'.format(features_suffix)), sparse_df)
			saveDict(length, SaveDir, 'Length-{:s}.json'.format(features_suffix))

		else:
			length = {}
			sparse_df = sparse.csr_matrix(sparse.load_npz(os.path.join(SaveDir, 'X.npz')))
			for agent in active_features:
				f = getFeaturesSuffix([agent])
				single = sparse.csr_matrix(sparse.load_npz(os.path.join(SaveDir, 'X-{:s}.npz'.format(f))))
				sparse_df = sparse.hstack([sparse_df,single[:,4:]]).tocsr()
				length[agent] = loadDict(SaveDir,'Length-{:s}.json'.format(f))
			sparse.save_npz(os.path.join(SaveDir, 'X-{:s}.npz'.format(features_suffix)), sparse_df)
			saveDict(length, SaveDir, 'Length-{:s}.json'.format(features_suffix))

		return sparse_df, length

	def loadKTMData(self, dataset_params, all_features):
		def to_dataset(data):
		    data = tf.ragged.constant(data)
		    data = tf.data.Dataset.from_tensor_slices(data)
		    data = data.map(lambda x: x)
		    return data
		def get_data_size(dataset):
		    if tf.__version__ == "2.2.0":
		        data_size = tf.data.experimental.cardinality(dataset).numpy()
		    else:
		        data_size = dataset.cardinality().numpy()
		    return data_size
		def split_dataset(dataset, train_fraction):
			data_size = get_data_size(dataset)
			train_size = int(data_size * train_fraction)
			train_size = train_size
			test_size = data_size - train_size
			train_dataset = dataset.take(train_size)
			test_dataset = dataset.skip(train_size)
			return train_dataset, test_dataset
		active = dataset_params['active']
		window_lengths = dataset_params['window_lengths'] 
		batch_size = dataset_params['batch_size']
		train_fraction = dataset_params['train_fraction']
		
		raw_data, length = self.loadSparseDF(active_features=active, window_lengths=window_lengths, all_features=all_features)
		raw_data = raw_data.toarray()
		data = to_dataset(raw_data[:, 4:])
		label = to_dataset(raw_data[:, 3])
		dataset = tf.data.Dataset.zip((data, label))
		dataset = dataset.map(lambda data, label: (tf.cast(data, dtype=tf.float32), tf.cast(label, dtype=tf.float32)))
		train_dataset, test_dataset = split_dataset(dataset, train_fraction=train_fraction)
		train_dataset = train_dataset.padded_batch(batch_size, drop_remainder=False)
		test_dataset = test_dataset.padded_batch(batch_size, drop_remainder=False)
		return train_dataset, test_dataset

	def getExtraStatics(self):

		if os.path.exists(os.path.join(self.LCDataDir, 'ExtraStaticInformation.json')):
			StaticInformation = loadDict(self.LCDataDir,'ExtraStaticInformation.json')
		ExtraStaticInformation = {}

		[df, QMatrix, StaticInformation, DictList] = self.dataprocessor.loadLCData()
		QMatrix = QMatrix.toarray()

		# Transform q-matrix into dictionary
		dict_q_mat = {i:set() for i in range(QMatrix.shape[0])}
		for elt in np.argwhere(QMatrix == 1):
			dict_q_mat[elt[0]].add(elt[1])

		i = 0
		userNum = len(df['user_id'].unique())
		print("userNum:", userNum)

		window_lengths = [3600*24*30*365]
		q_kc = defaultdict(lambda: OurQueue(window_lengths))  # Prepare counters for time windows kc_related
		SkillDelay = 0.0
		SkillDelayNum = 0.0
		StudyPeriod = 0.0

		for stud_id in df['user_id'].unique():
			df_stud = df[df['user_id']==stud_id][['user_id', 'item_id', 'timestamp', 'correct']].copy()
			df_stud.sort_values(by='timestamp', inplace=True) # Sort values 
			StudyPeriod += (df_stud['timestamp'].max()-df_stud['timestamp'].min())
			df_stud = np.array(df_stud)

			skills = QMatrix[df_stud[:,1].astype(int)].copy()

			for l, (item_id, t, correct) in enumerate(zip(df_stud[:,1], df_stud[:,2], df_stud[:,3])):
				for skill_id in dict_q_mat[item_id]:
					pret = q_kc[stud_id, skill_id].get_last()
					if pret!=-1e8:
						SkillDelay += (t-pret)
						SkillDelayNum += 1
					q_kc[stud_id, skill_id].push(t)

			i+=1
			if i%100 == 0:
				print(i,userNum)

		ExtraStaticInformation['MeanSkillDelay'] = SkillDelay/SkillDelayNum
		ExtraStaticInformation['MeanStudyPeriod'] = StudyPeriod/userNum
		ExtraStaticInformation['dastSparsity'] = len(df.drop_duplicates(subset=['user_id', 'item_id']))/StaticInformation['userNum']*StaticInformation['itemNum']

		saveDict(ExtraStaticInformation,self.LCDataDir,'ExtraStaticInformation.json')
		return ExtraStaticInformation

if __name__ == "__main__":
	#algebra08原始数据里的最值
	#low_time = "2008-09-08 14:46:48"
	#high_time = "2009-07-06 18:02:12"
    '''
	isTest = True
	isAll = False

	datasetName = 'algebra08'

	if datasetName == 'algebra08':
		if isTest == True:
			userLC = [10, 3000]
			problemLC = [10, 5000]
			low_time = "2008-12-21 14:46:48"
			high_time = "2009-01-01 00:00:00"
			timeLC = [low_time, high_time]
		else:
			userLC = [30, 3600]
			problemLC = [30, 1e9]
			low_time = "2008-09-08 14:46:48"
			high_time = "2009-01-01 00:00:00"
			timeLC = [low_time, high_time]
		if isAll == True:
			userLC = [10, 1e9]
			problemLC = [10, 1e9]
			low_time = "2008-09-08 14:46:48"
			high_time = "2009-07-06 18:02:12"
			timeLC = [low_time, high_time]


	elif datasetName == 'algebra05':
		if isTest == True:
			userLC = [10, 3000]
			problemLC = [10, 5000]
			low_time = "2006-06-01 00:00:00"
			high_time = "2006-06-07 11:12:38"
			timeLC = [low_time, high_time]
		else:
			userLC = [10, 1e9]
			problemLC = [10, 1e9]
			low_time = "2005-08-30 09:50:35"
			high_time = "2006-06-07 11:12:38"
			timeLC = [low_time, high_time]

	elif datasetName == 'bridge_algebra06':
		if isTest == True:
			userLC = [10, 3000]
			problemLC = [10, 5000]
			low_time = "2006-06-10 08:26:16"
			high_time = "2007-06-20 13:36:57"
			timeLC = [low_time, high_time]
		else:
			userLC = [10, 1e9]
			problemLC = [10, 1e9]
			low_time = "2006-10-05 08:26:16"
			high_time = "2007-06-20 13:36:57"
			timeLC = [low_time, high_time]

	a = _DataProcessor(userLC, problemLC, timeLC, 'kdd', datasetName = datasetName, TmpDir = './data')

	[df, QMatrix, StaticInformation, DictList] = a.dataprocessor.loadLCData()
	print(df.columns)
	print('**************StaticInformation**************')
	printDict(StaticInformation)
	printDict(a.LC_params)

	kFold = 5
	df_length, dict_data = a.getLengthDistribution(kFold)
    '''

    '''
	Features = {}
	Features['users'] = True #用于das3h中特征
	Features['items'] = True #用于das3h中特征
	Features['skills'] = True
	Features['wins'] = True
	Features['fails'] = False
	Features['attempts'] = True
	Features2 = {}
	Features2['tw_kc'] = True
	Features2['tw_items'] = False
	all_features = ['users', 'items', 'skills', 'wins', 'fails', 'attempts']
	active_features = [key for key, value in Features.items() if value]
	all_features = ['users', 'items', 'skills', 'wins', 'fails', 'attempts']
	active_features = [key for key, value in Features.items() if value]

	features_suffix = ''.join([features[0] for features in active_features])
	if Features["tw_kc"]:
		features_suffix += 't1'
		tw = "tw_kc"
	elif Features["tw_items"]:
		features_suffix += 't2'
		tw = "tw_items"
	else:
		tw = None

	sparsedf, train, test = a.loadDAS3HData(active_features, features_suffix, 0.8, tw=tw)
	printDict(a.getExtraStatics())
	'''

	#hdu原始数据里的最值
	#low_time = "2018-06-01 00:00:00" 
	#high_time = "2018-11-29 00:00:00"

    '''
	isTest = True
	if isTest == True:
        userLC = [10, 500, 0.1, 1]
        problemLC = [10, 500, 0, 1]
        low_time = "2018-11-22 00:00:00"
        high_time = "2018-11-29 00:00:00"
        timeLC = [low_time, high_time]
    else:
        userLC = [30, 3600, 0.1, 1]
        problemLC = [30, 1e9, 0, 1]
        low_time = "2018-06-01 00:00:00"
        high_time = "2018-11-29 00:00:00"
        timeLC = [low_time, high_time]
	a = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir = '../data')
	[df, QMatrix, StaticInformation, DictList] = a.dataprocessor.loadLCData()
	print('**************StaticInformation**************')
	printDict(StaticInformation)
	'''

    '''
	isTest = True
    isAll = False
    if isTest == True:
        userLC = [10, 300]
        problemLC = [10, 300]
        low_time = "2012-09-01 00:00:00"
        high_time = "2012-09-10 00:00:00"
        timeLC = [low_time, high_time]
    else:
        userLC = [10, 3000]
        problemLC = [10, 3000]
        low_time = "2012-09-01 00:00:00"
        high_time = "2012-09-30 00:00:00"
        timeLC = [low_time, high_time]
    if isAll == True:
        userLC = [10, 1e9]
        problemLC = [10, 1e9]
        low_time = "2012-09-01 00:00:00"
        high_time = "2013-09-01 00:00:00"
        timeLC = [low_time, high_time]
	a = _DataProcessor(userLC, problemLC, timeLC, 'assist', TmpDir = '../data')
	[df, QMatrix, StaticInformation, DictList] = a.dataprocessor.loadLCData()
	print('**************StaticInformation**************')
	printDict(StaticInformation)
	'''

    '''
	test loadSparseDF
	'''

    '''
	Features = {}
	Features['users'] = False #用于das3h中特征
	Features['items'] = False #用于das3h中特征
	Features['skills'] = False
	Features['lasttime_1kc'] = True
	Features['lasttime_2items'] = False
	Features['lasttime_3sequence'] = False
	Features['interval_1kc'] = False
	Features['interval_2items'] = False
	Features['interval_3sequence'] = False
	Features['wins_1kc'] = True
	Features['wins_2items'] = False
	Features['fails_1kc'] = False
	Features['fails_2items'] = False
	Features['attempts_1kc'] = True
	Features['attempts_2items'] = False

	window_lengths = [3600 * 24 * 30 * 12 * 100]

	active_features = [key for key, value in Features.items() if value]
	all_features = list(Features.keys())
	sparse_df, Length = a.loadSparseDF(active_features, window_lengths, all_features)
	print('**************sparse_df**************')
	print(sparse_df.shape)
	printDict(Length)
	'''
