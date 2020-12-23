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

def DAS3H(a, active, window_lengths, isKfold, model_params, FM_params):

	prefix = ''
	if set(active) == {'users', 'items'} and model_params['dim'] == 0:
		prefix = 'IRT'
	elif set(active) == {'users', 'items'} and model_params['dim'] > 0:
		prefix = 'MIRTb'
	elif set(active) == {'items', 'skills', 'wins_3das3h', 'fails_3das3h'}:
		prefix = 'KTM'
	elif set(active) == {'skills', 'attempts_3das3h'}:
		prefix = 'AFM'
	elif set(active) == {'skills', 'wins_3das3h', 'fails_3das3h'}:
		prefix = 'PFA'
	elif set(active) == {'users', 'items', 'skills', 'wins_4das3hkc', 'attempts_4das3hkc'}:
		prefix = 'DAS3H'
	elif set(active) == {'users', 'items', 'wins_5das3hitems', 'attempts_5das3hitems'}:
		prefix = 'DASH'
	else:
		prefix = 'TEST_' + features_suffix

	X, Length = a.loadSparseDF(active, window_lengths)
	y = X[:,3].toarray().flatten()

	[df, QMatrix, StaticInformation, DictList] = a.dataprocessor.loadLCData()

	saveDir = os.path.join(a.LCDataDir, 'das3h', prefix+'_isKfold_'+str(isKfold)[0])
	prepareFolder(saveDir)

	y_tests = {}
	y_pred_tests = {}

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
				model.fit(X_train[:,4:], y_train) # the 5 first columns are the non-sparse dataset
				y_pred_test = model.predict_proba(X_test[:,4:])[:, 1]
			else:
				fm = pywFM.FM(**FM_params)
				model = fm.run(X_train[:,4:], y_train, X_test[:,4:], y_test)
				y_pred_test = np.array(model.predictions)
				model.rlog.to_csv(os.path.join(saveDir, str(run_id), 'rlog.csv'))

			y_tests[run_id] = y_test
			y_pred_tests[run_id] = y_pred_test
	else:
		prepareFolder(os.path.join(saveDir, str(0)))

		users_train, users_test = a.loadDAS3HData(model_params['trainRate'])
		X_train = X[np.where(np.isin(X[:,0].toarray().flatten(),users_train))]
		y_train = X_train[:,3].toarray().flatten()
		X_test = X[np.where(np.isin(X[:,0].toarray().flatten(),users_test))]
		y_test = X_test[:,3].toarray().flatten()

		if model_params['dim'] == 0:
			print('fitting...')
			model = LogisticRegression(solver="newton-cg", max_iter=400)
			model.fit(X_train[:,4:], y_train) # the 5 first columns are the non-sparse dataset
			y_pred_test = model.predict_proba(X_test[:,4:])[:, 1]
		else:
			fm = pywFM.FM(**FM_params)
			model = fm.run(X_train[:,4:], y_train, X_test[:,4:], y_test)
			y_pred_test = np.array(model.predictions)
			model.rlog.to_csv(os.path.join(saveDir, str(0), 'rlog.csv'))

		y_tests[0] = y_test
		y_pred_tests[0] = y_pred_test

	return y_tests, y_pred_tests, saveDir


def runKDD(active, window_lengths, isTest, isKfold, metrics1, metrics2, metrics_tf1, metrics_tf2, TmpDir):
	features_suffix = getFeaturesSuffix(active)

 
	#######################################
	# model parameters
	#######################################
	model_params = {
		'dim': 5,
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


	#######################################
	# LC parameters
	#######################################
	#algebra08原始数据里的最值
	#low_time = "2008-09-08 14:46:48"
	#high_time = "2009-07-06 18:02:12"

	if isTest == True:
		userLC = [10, 3000]
		problemLC = [10, 5000]
		low_time = "2008-12-21 14:46:48"
		high_time = "2009-01-01 00:00:00"
		timeLC = [low_time, high_time]
	else:
		userLC = [10, 3000]
		problemLC = [10, 5000]
		low_time = "2008-09-08 14:46:48"
		high_time = "2009-07-06 18:02:12"
		timeLC = [low_time, high_time]

	a = _DataProcessor(userLC, problemLC, timeLC, 'kdd', TmpDir = TmpDir)

	y_tests, y_pred_tests, saveDir = DAS3H(a, active, window_lengths, isKfold, model_params, FM_params)


	results={'LC_params':a.LC_params,'model_params':model_params,'FM_params':FM_params,'results':{}}

	for run_id in range(len(y_tests)):
		y_test = y_tests[run_id]
		y_pred_test = y_pred_tests[run_id]

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

	saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')

def runOJ(active, window_lengths, isTest, isKfold, metrics1, metrics2, metrics_tf1, metrics_tf2, TmpDir, FM_params):
	features_suffix = getFeaturesSuffix(active)
 
	#######################################
	# model parameters
	#######################################
	model_params = {
		'dim': 0,
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


	#######################################
	# LC parameters
	#######################################
	#hdu原始数据里的最值，可以注释，不要删
	#low_time = "2018-06-01 00:00:00" 
	#high_time = "2018-11-29 00:00:00"
	isTest = False

	if isTest == True:
		userLC = [10, 500, 0.1, 1]
		problemLC = [10, 500, 0, 1]
		low_time = "2018-11-22 00:00:00"
		high_time = "2018-11-29 00:00:00"
		timeLC = [low_time, high_time]
	else:
		userLC = [10, 500, 0.1, 1]
		problemLC = [10, 500, 0, 1]
		low_time = "2018-06-01 00:00:00"
		high_time = "2018-11-29 00:00:00"
		timeLC = [low_time, high_time]

	a = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir = TmpDir)

	y_tests, y_pred_tests, saveDir = DAS3H(a, active, window_lengths, isKfold, model_params)


	results={'LC_params':a.LC_params,'model_params':model_params,'FM_params':FM_params,'results':{}}

	for run_id in range(len(y_tests)):
		y_test = y_tests[run_id]
		y_pred_test = y_pred_tests[run_id]

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

	saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')

def runAssist(active, window_lengths, isTest, isKfold, metrics1, metrics2, metrics_tf1, metrics_tf2, TmpDir, FM_params):
	features_suffix = getFeaturesSuffix(active)

	#######################################
	# model parameters
	#######################################
	model_params = {
		'dim': 0,
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


	#######################################
	# LC parameters
	#######################################
	#assistments12原始数据里的最值，可以注释，不要删
	#low_time = "2012-09-01 00:00:00"
	#high_time = "2013-09-01 00:00:00"
	isTest = True
	if isTest == True:
		userLC = [10, 3000]
		problemLC = [10, 3000]
		low_time = "2012-09-01 00:00:00"
		high_time = "2012-09-30 00:00:00"
		timeLC = [low_time, high_time]
	else:
		userLC = [10, 3000]
		problemLC = [10, 3000]
		low_time = "2012-09-01 00:00:00"
		high_time = "2013-01-01 00:00:00"
		timeLC = [low_time, high_time]
	
	a = _DataProcessor(userLC, problemLC, timeLC, 'assist', TmpDir = TmpDir)

	y_tests, y_pred_tests, saveDir = DAS3H(a, active, window_lengths, isKfold, model_params)


	results={'LC_params':a.LC_params,'model_params':model_params,'FM_params':FM_params,'results':{}}

	for run_id in range(len(y_tests)):
		y_test = y_tests[run_id]
		y_pred_test = y_pred_tests[run_id]

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

	saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')

if __name__ == "__main__":
	Features = {}
	Features['users'] = True #用于das3h中特征
	Features['items'] = True #用于das3h中特征
	Features['skills'] = False
	Features['lasttime_0kcsingle'] = False
	Features['lasttime_1kc'] = False
	Features['lasttime_2items'] = False
	Features['lasttime_3sequence'] = False
	Features['interval_1kc'] = False
	Features['interval_2items'] = False
	Features['interval_3sequence'] = False
	Features['wins_1kc'] = False
	Features['wins_2items'] = False
	Features['wins_3das3h'] = False #用于das3h中特征
	Features['wins_4das3hkc'] = False #用于das3h中特征
	Features['wins_5das3hitems'] = False #用于das3h中特征
	Features['fails_1kc'] = False
	Features['fails_2items'] = False
	Features['fails_3das3h'] = False
	Features['attempts_1kc'] = False 
	Features['attempts_2items'] = False
	Features['attempts_3das3h'] = False #用于das3h中特征
	Features['attempts_4das3hkc'] = False #用于das3h中特征
	Features['attempts_5das3hitems'] = False #用于das3h中特征

	window_lengths = [3600*24*30*365]
	#window_lengths = [3600 * 1e19, 3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]

	active_features = [key for key, value in Features.items() if value]

	isTest = True
	isKfold = False
	# TmpDir = "./DataProcessor/data"
	TmpDir = "./data"

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

	runKDD(active_features, window_lengths, isTest, isKfold, metrics1, metrics2, metrics_tf1, metrics_tf2, TmpDir)