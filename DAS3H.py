import os
import sys
import json
import pywFM
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from scipy.sparse import load_npz, hstack, csr_matrix

sys.path.append("./DataProcessor/")

from public import *
from DataProcessor import _DataProcessor
  

# Location of libFM's compiled binary file
os.environ['LIBFM_PATH'] = '~/libfm/bin/'

def runOJ():
	Features = {}
	Features['users'] = True
	Features['items'] = True
	Features['skills'] = False
	Features['lasttime_0kcsingle'] = False
	Features['lasttime_1kc'] = False
	Features['lasttime_2items'] = False
	Features['lasttime_3sequence'] = False
	Features['wins_1kc'] = False
	Features['wins_2items'] = False
	Features['wins_3das3h'] = False #用于das3h中特征
	Features['wins_4das3hkc'] = False #用于das3h中特征
	Features['wins_5das3hitems'] = False #用于das3h中特征
	Features['fails'] = False
	Features['attempts_1kc'] = False
	Features['attempts_2items'] = False
	Features['attempts_3das3h'] = False #用于das3h中特征
	Features['attempts_4das3hkc'] = False #用于das3h中特征
	Features['attempts_5das3hitems'] = False #用于das3h中特征

	active = [key for key, value in Features.items() if value]
	features_suffix = getFeaturesSuffix(active)

	window_lengths = [3600]

 
	#######################################
	# model parameters
	#######################################
	model_params = {
		'dim': 5,
		'kFold': 5,
		'iter': 300,
		'threshold': 0.5,
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
	userLC = [10,500,0.1,1]
	problemLC = [10,500,0,1]
	#hdu原始数据里的最值，可以注释，不要删
	low_time = "2018-06-01 00:00:00" 
	high_time = "2018-11-29 00:00:00"
	timeLC = [low_time, high_time]

	a = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir = "./DataProcessor/data")
	LC_params = a.LC_params

	prefix = ''
	if set(active) == {'users', 'items'} and model_params['dim'] == 0:
		prefix = 'IRT'
	elif set(active) == {'users', 'items'} and model_params['dim'] > 0:
		prefix = 'MIRTb'
	elif set(active) == {'skills', 'attempts'}:
		prefix = 'AFM'
	elif set(active) == {'skills', 'wins', 'fails'}:
		prefix = 'PFA'
	elif set(active) == {'users', 'items', 'skills', 'wins', 'attempts', 'tw_kc'}:
		prefix = 'DAS3H'
	elif set(active) == {'users', 'items', 'wins', 'attempts', 'tw_items'}:
		prefix = 'DASH'
	else:
		prefix = 'TEST_' + features_suffix

	LCDataDir = a.LCDataDir
	saveDir = os.path.join(LCDataDir, 'das3h', prefix)
	prepareFolder(saveDir)
	for run_id in range(model_params['kFold']):
	   prepareFolder(os.path.join(saveDir, str(run_id)))
	X, Length = a.loadSparseDF(active, window_lengths)
	y = X[:,3].toarray().flatten()

	[df, QMatrix, StaticInformation, DictList] = a.dataprocessor.loadLCData()
	dict_data = a.loadSplitInfo(model_params['kFold'])

	results={'LC_params':LC_params,'model_params':model_params,'FM_params':FM_params,'results':{}}
	metrics_tf = {'tf_Accuracy':tf.keras.metrics.Accuracy(),
				'tf_Precision':tf.keras.metrics.Precision(thresholds=model_params['threshold']),
				'tf_Recall':tf.keras.metrics.Recall(thresholds=model_params['threshold']),
			   'tf_MSE':tf.keras.metrics.MeanSquaredError(),
			   'tf_MAE':tf.keras.metrics.MeanAbsoluteError(),
			   'tf_RMSE':tf.keras.metrics.RootMeanSquaredError(),
			   'tf_AUC':tf.keras.metrics.AUC(),
			   'tf_AUC_1000': tf.keras.metrics.AUC(num_thresholds=1000)
	}

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

		results['results'][run_id] = {}
		temp = results['results'][run_id]

		for metric in metrics_tf:
			m = metrics[metric]
			m.reset_states()
			m.update_state(y_test, y_pred_test)
			temp[metric] = m.result().numpy()

	saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')

if __name__ == "__main__":
	runOJ()