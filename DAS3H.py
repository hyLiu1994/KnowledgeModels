import os
import sys
import json
import pywFM
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf 
from sklearn import metrics
from tensorflow.keras import layers
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz, hstack, csr_matrix

sys.path.append("./DataProcessor/")

from public import *
from DataProcessor import _DataProcessor
  

# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = '~/libfm/bin/'

def DAS3H(a, active, tw, isKfold, model_params):

	dim = model_params['dim']

	# FM parameters
	FM_params = {
		'task': 'classification',
		'num_iter': model_params['iter'],
		'rlog': True,
		'learning_method': 'mcmc',
		'k2': dim
	}

	print(active)
	prefix = ''
	if set(active) == {'users', 'items'} and dim == 0:
		prefix = 'IRT'
	elif set(active) == {'users', 'items'} and dim > 0:
		prefix = 'MIRTb'
	elif set(active) == {'skills', 'attempts'}:
		prefix = 'AFM'
	elif set(active) == {'skills', 'wins', 'fails'}:
		prefix = 'PFA'
	elif set(active) == {'items', 'skills', 'wins', 'fails'}:
		prefix = 'KTM'
	elif set(active) == {'users', 'items', 'skills', 'wins', 'attempts'} and ( tw == 'tw_kc'):
		prefix = 'DAS3H'
	elif set(active) == {'users', 'items', 'wins', 'attempts'} and ( tw == 'tw_items'):
		prefix = 'DASH'
	else:
		for f in active:
			prefix += f[0]
		if tw == 'tw_kc':
			prefix += 't1'
		else:
			prefix += 't2'
	print(prefix)


	[df, QMatrix, StaticInformation, DictList] = a.dataprocessor.loadLCData()
	X, dict_data = a.loadDAS3HData(active, features_suffix, 0.8, tw=tw)
	y = X[:,3].toarray().flatten()

	saveDir = os.path.join(a.LCDataDir, 'das3h', 'results_K'+str(isKfold)[0], prefix)
	prepareFolder(saveDir)

	metrics1 = {'MAE':metrics.mean_absolute_error,
	'MSE':metrics.mean_squared_error,
	'AUC':metrics.roc_auc_score,
	}

	metrics2 = {'Accuracy':metrics.accuracy_score,
	'Precision':metrics.precision_score,
	'AP':metrics.average_precision_score,
	'Recall':metrics.recall_score,
	'F1-score':metrics.f1_score,
	}

	metrics_tf1 = {'tf_Accuracy':tf.keras.metrics.Accuracy(),
	}

	metrics_tf2 = {'tf_Precision':tf.keras.metrics.Precision(thresholds = 0.5),
	'tf_Recall':tf.keras.metrics.Recall(thresholds = 0.5),
	'tf_MSE':tf.keras.metrics.MeanSquaredError(),
	'tf_MAE':tf.keras.metrics.MeanAbsoluteError(),
	'tf_RMSE':tf.keras.metrics.RootMeanSquaredError(),
	'tf_AUC':tf.keras.metrics.AUC(),
	'tf_AUC_1000': tf.keras.metrics.AUC(num_thresholds=1000)
	}


	results={'LC_params':a.LC_params,'model_params':model_params,'results':{}}

	if isKfold:
		for run_id in range(model_params['kFold']):
			prepareFolder(os.path.join(saveDir, str(run_id)))
			dict_data = a.loadSplitInfo(model_params['kFold'])

		for run_id in range(model_params['kFold']):
			users_train = dict_data[str(run_id)]['train']
			users_test = dict_data[str(run_id)]['test']

			X_train = X[np.where(np.isin(X[:,0].toarray().flatten(),users_train))]
			y_train = X_train[:,3].toarray().flatten()
			X_test = X[np.where(np.isin(X[:,0].toarray().flatten(),users_test))]
			y_test = X_test[:,3].toarray().flatten()

			if model_params['dim'] == 0:
				print('fitting...')
				model = LogisticRegression(solver="newton-cg", max_iter=400)
				model.fit(X_train[:,5:], y_train) # the 5 first columns are the non-sparse dataset
				y_pred_test = model.predict_proba(X_test[:,5:])[:, 1]
			else:
				fm = pywFM.FM(**FM_params)
				model = fm.run(X_train[:,5:], y_train, X_test[:,5:], y_test)
				y_pred_test = np.array(model.predictions)
				model.rlog.to_csv(os.path.join(saveDir, str(run_id), 'rlog.csv'))

			results['results'][run_id] = {}
			temp = results['results'][run_id]
			for metric in metrics1:
				temp[metric] = metrics1[metric](y_test, y_pred_test)

			for metric in metrics2:
				temp[metric] = metrics2[metric](y_test, (y_pred_test>0.5).astype(int))
				
			for metric in metrics_tf1:
				m = metrics_tf1[metric]
				m.reset_states()
				m.update_state(y_test, tf.greater_equal(y_pred_test,0.5))
				temp[metric] = m.result().numpy()
				
			for metric in metrics_tf2:
				m = metrics_tf2[metric]
				m.reset_states()
				m.update_state(y_test, y_pred_test)
				temp[metric] = m.result().numpy()
	else:
		X_train = X[np.where(np.isin(X[:,0].toarray().flatten(),dict_data['0']['train']))]
		y_train = X_train[:,3].toarray().flatten()
		X_test = X[np.where(np.isin(X[:,0].toarray().flatten(),dict_data['0']['test']))]
		y_test = X_test[:,3].toarray().flatten()

		if model_params['dim'] == 0:
			print('fitting...')
			model = LogisticRegression(solver="newton-cg", max_iter=model_params['iter'])
			model.fit(X_train[:,4:], y_train) # the 5 first columns are the non-sparse dataset
			y_pred_test = model.predict_proba(X_test[:,4:])[:, 1]
		else:
			fm = pywFM.FM(**FM_params)
			model = fm.run(X_train[:,4:], y_train, X_test[:,4:], y_test)
			y_pred_test = np.array(model.predictions)
			model.rlog.to_csv(os.path.join(saveDir, 'rlog'+getLegend(model_params)+'.csv'))

		temp = results['results']
		for metric in metrics1:
			temp[metric] = metrics1[metric](y_test, y_pred_test)

		for metric in metrics2:
			temp[metric] = metrics2[metric](y_test, (y_pred_test>0.5).astype(int))
				
		for metric in metrics_tf1:
			m = metrics_tf1[metric]
			m.reset_states()
			m.update_state(y_test, tf.greater_equal(y_pred_test,0.5))
			temp[metric] = m.result().numpy()
				
		for metric in metrics_tf2:
			m = metrics_tf2[metric]
			m.reset_states()
			m.update_state(y_test, y_pred_test)
			temp[metric] = m.result().numpy()
	saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')
	return results


def runKDD(datasetName, isTest = True, isAll = False, TmpDir = "./data"):

	#######################################
	# LC parameters
	#######################################
	#algebra08原始数据里的最值
	#low_time = "2008-09-08 14:46:48"
	#high_time = "2009-07-06 18:02:12"

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

	a = _DataProcessor(userLC, problemLC, timeLC, 'kdd', datasetName = datasetName, TmpDir = TmpDir)

	return a


def runOJ(datasetName = 'hdu', isTest = True, isAll = False, TmpDir = "./data"):
 
	#######################################
	# LC parameters
	#######################################
	#hdu原始数据里的最值，可以注释，不要删
	#low_time = "2018-06-01 00:00:00" 
	#high_time = "2018-11-29 00:00:00"

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

	a = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir = TmpDir)
	return a

def runAssist(datasetName = 'assistments12', isTest = True, isAll = False, TmpDir = "./data"):
	#######################################
	# LC parameters
	#######################################
	#assistments12原始数据里的最值，可以注释，不要删
	#low_time = "2012-09-01 00:00:00"
	#high_time = "2013-09-01 00:00:00"
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

	a = _DataProcessor(userLC, problemLC, timeLC, 'assist', TmpDir = TmpDir)
	return a

if __name__ == "__main__":
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

	features_suffix = ''.join([features[0] for features in active_features])
	if Features2["tw_kc"]:
		features_suffix += 't1'
		tw = "tw_kc"
	elif Features2["tw_items"]:
		features_suffix += 't2'
		tw = "tw_items"
	else:
		tw = None

	#######################################
	# model parameters
	#######################################
	model_params = {
		'dim': 20,
		'kFold': 5,
		'trainRate':0.8,
		'iter': 300,
		'active_features': features_suffix
	}
 

	# FM parameters
	FM_params = {
		'task': 'classification',
		'num_iter': model_params['iter'],
		'rlog': True,
		'learning_method': 'mcmc',
		'k2': model_params['dim']
	}
	'''


	# TmpDir = "./DataProcessor/data"
	TmpDir = "./data"
	isKfold = True

	# bridge_algebra06 algebra05
	a = runKDD(datasetName = 'bridge_algebra06', isTest = False, isAll = False, TmpDir = TmpDir)
	#a = runOJ(isTest = False, isAll = False, TmpDir = TmpDir)
	#a = runAssist(isTest = True, isAll = False, TmpDir = TmpDir)
	
	active_features = ['skills', 'attempts']
	features_suffix = 'sa'
	tw = None

	model_params = {
		'dim': 20,
		'kFold': 5,
		'trainRate':0.8,
		'iter': 300,
		'active_features': features_suffix
	}
 
	FM_params = {
		'task': 'classification',
		'num_iter': model_params['iter'],
		'rlog': True,
		'learning_method': 'mcmc',
		'k2': model_params['dim']
	}

	results = DAS3H(a, active_features, tw, isKfold, model_params)
	printDict(results['results'])

	active_features = ['skills', 'wins', 'fails']
	features_suffix = 'swf'
	tw = None
	model_params = {
		'dim': 20,
		'kFold': 5,
		'trainRate':0.8,
		'iter': 300,
		'active_features': features_suffix
	}
	results = DAS3H(a, active_features, tw, isKfold, model_params)
	printDict(results['results'])

	active_features = ['items', 'skills', 'wins', 'fails']
	features_suffix = 'iswf'
	tw = None
	model_params = {
		'dim': 20,
		'kFold': 5,
		'trainRate':0.8,
		'iter': 300,
		'active_features': features_suffix
	}
	results = DAS3H(a, active_features, tw, isKfold, model_params)
	printDict(results['results'])

	active_features = ['users', 'items', 'skills', 'wins', 'attempts']
	features_suffix = 'uiswat1'
	tw = 't1'
	model_params = {
		'dim': 20,
		'kFold': 5,
		'trainRate':0.8,
		'iter': 300,
		'active_features': features_suffix
	}
	results = DAS3H(a, active_features, tw, isKfold, model_params)
	printDict(results['results'])

	active_features = ['users', 'items', 'wins', 'attempts']
	features_suffix = 'uiwat2'
	tw = 't2'
	model_params = {
		'dim': 20,
		'kFold': 5,
		'trainRate':0.8,
		'iter': 300,
		'active_features': features_suffix
	}
	results = DAS3H(a, active_features, tw, isKfold, model_params)
	printDict(results['results'])