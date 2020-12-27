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
# import tensorflow_probability as tfp


class KTM(tf.keras.Model):
    def __init__(self, model_params):
        super(KTM, self).__init__()
        feature_num = model_params['feature_num']
        embed_dim = model_params['embed_dim']
        self.metrics_path = 'file://' + model_params['metrics_path']
        self.threshold = model_params['threshold'] 
        self.embed = self.add_weight(name="embed", shape=(feature_num, embed_dim), initializer="uniform")
        self.bias = self.add_weight(name="bias", shape=(feature_num, 1), initializer="zeros")
        self.global_bias = self.add_weight(name="global_bias", shape=(1, 1), initializer="zeros")
        # link
        # probit
        # self.link_fun = tfp.distributions.Normal(loc=0., scale=1.).cdf
        # sigmoid
        self.link_fun = tf.keras.activations.sigmoid
        # train
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.opti = tf.keras.optimizers.Adam()
        # metrics
        self.metrics_format = "epoch,time,train_loss,train_acc,train_pre,train_rec,train_auc,train_mae,train_rmse,test_loss,test_acc,test_pre,test_rec,test_auc,test_mae,test_rmse"
        # metrics
        self.metrics_loss = tf.keras.metrics.BinaryCrossentropy()
        self.metrics_acc = tf.keras.metrics.BinaryAccuracy(threshold = self.threshold)
        self.metrics_pre = tf.keras.metrics.Precision(thresholds = self.threshold)
        self.metrics_rec = tf.keras.metrics.Recall(thresholds = self.threshold)
        self.metrics_auc = tf.keras.metrics.AUC()
        self.metrics_mae = tf.keras.metrics.MeanAbsoluteError()
        self.metrics_rmse = tf.keras.metrics.RootMeanSquaredError()

    def call(self, data):
        data = tf.expand_dims(data, axis=-1)
        embed = tf.expand_dims(self.embed, axis=0)
        embed = tf.multiply(data, embed)
        dot = tf.matmul(embed, embed, transpose_b=True)
        dim = tf.shape(dot)[-1]
        up_tri = 1 - tf.linalg.band_part(tf.ones((dim, dim)),-1,0)
        selected_dot = tf.multiply(dot, up_tri)
        dot = tf.reduce_sum(tf.reduce_sum(dot, axis=-1), axis=-1)
        bias = tf.expand_dims(self.bias, axis=0)
        bias = tf.multiply(data, bias)
        bias = tf.squeeze(bias, axis=-1)
        bias = tf.reduce_sum(bias, axis=-1)
        global_bias = tf.squeeze(tf.squeeze(self.global_bias, axis=-1), axis=-1)
        pred = global_bias + bias + dot
        pred = self.link_fun(pred)
        return pred

    @tf.function(experimental_relax_shapes=True)
    def metrics_step(self, data, label):
        pred = self(data)
        self.loss_function(pred, label)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data, label):
        with tf.GradientTape() as tape:
            pred = self(data)
            label = tf.expand_dims(label, axis=-1)
            pred = tf.expand_dims(pred, axis=-1)
            loss = self.loss_function(label, pred)
            self.update_metrics(label, pred) 
        grad = tape.gradient(loss, self.trainable_variables)
        self.opti.apply_gradients(zip(grad, self.trainable_variables))
    
    def resetMetrics(self):
        self.metrics_loss.reset_states()
        self.metrics_acc.reset_states()
        self.metrics_pre.reset_states()
        self.metrics_rec.reset_states()
        self.metrics_auc.reset_states()
        self.metrics_mae.reset_states()
        self.metrics_rmse.reset_states()
    
    def update_metrics(self, label, pred):
        self.metrics_loss.update_state(label, pred)
        self.metrics_acc.update_state(label, pred)
        self.metrics_pre.update_state(label, pred)
        self.metrics_rec.update_state(label, pred)
        self.metrics_auc.update_state(label, pred)
        self.metrics_mae.update_state(label, pred)
        self.metrics_rmse.update_state(label, pred)

    def loss_function(self, label, pred):
        loss = self.loss(label, pred)
        self.update_metrics(label, pred)
        return loss
        
@tf.function
def test(model, dataset):
    for data, label in dataset:
        model.metrics_step(data, label)

@tf.function
def train(epoch, model, train_dataset, test_dataset):
    element_num = tf.data.experimental.cardinality(train_dataset)
    tf.print(model.metrics_format, output_stream=model.metrics_path)
    start = tf.timestamp()
    for i, (data, label) in train_dataset.repeat(epoch).enumerate():
        model.train_step(data, label)
        if tf.equal(tf.math.floormod(i+1, element_num), 0):
            end = tf.timestamp()
            train_loss, train_acc, train_pre, train_rec, train_auc, train_mae, train_rmse = model.metrics_loss.result(),  model.metrics_acc.result(),  model.metrics_pre.result(), model.metrics_rec.result(), model.metrics_auc.result(), model.metrics_mae.result(), model.metrics_rmse.result()
            model.resetMetrics()

            test(model, test_dataset)
            test_loss, test_acc, test_pre, test_rec, test_auc, test_mae, test_rmse = model.metrics_loss.result(),  model.metrics_acc.result(),  model.metrics_pre.result(), model.metrics_rec.result(), model.metrics_auc.result(), model.metrics_mae.result(), model.metrics_rmse.result()
            model.resetMetrics()

            tf.print(tf.math.floordiv(i+1, element_num),
                    end - start, 
                    train_loss, train_acc, train_pre, train_rec, train_auc, train_mae, train_rmse,
                    test_loss, test_acc, test_pre, test_rec, test_auc, test_mae, test_rmse,
                    sep=',', output_stream=model.metrics_path)

            tf.print("epoch: ", tf.math.floordiv(i+1, element_num),
                    "time: ", end - start, 
                    "train_loss: ", train_loss, "train_acc: ", train_acc, "train_pre: ", train_pre,
                    "train_rec: ", train_rec, "train_auc: ", train_auc, "train_mae: ", train_mae, "train_rmse: ", train_rmse,
                    "test_loss: ", test_loss,  "test_acc: ", test_acc,  "test_pre: ", test_pre,
                    "test_rec: ", test_rec, "test_auc: ", test_auc, "test_mae: ", test_mae, "test_rmse: ", test_rmse)
            start = tf.timestamp()

def runKDD():
    Features = {}
    Features['users'] = True
    Features['items'] = True
    Features['skills'] = True
    Features['lasttime_0kcsingle'] = False
    Features['lasttime_1kc'] = False
    Features['lasttime_2items'] = False
    Features['lasttime_3sequence'] = False
    Features['wins_1kc'] = False
    Features['wins_1kc'] = True
    Features['wins_2items'] = False
    Features['wins_3das3h'] = False #用于das3h中特征
    Features['wins_4das3hkc'] = False #用于das3h中特征
    Features['wins_5das3hitems'] = False #用于das3h中特征   
    Features['fails_1kc'] = False
    Features['fails_1kc'] = True
    Features['attempts_1kc'] = False
    Features['attempts_2items'] = False
    Features['attempts_3das3h'] = False #用于das3h中特征
    Features['attempts_4das3hkc'] = False #用于das3h中特征
    Features['attempts_5das3hitems'] = False #用于das3h中特征
    active = [key for key, value in Features.items() if value]
    features_suffix = getFeaturesSuffix(active)
    window_lengths = [365 * 24 * 3600]
    ####################################### 
    # LC parameters
    #######################################
    userLC = [10,3000]
    problemLC = [10,5000]
    # algebra08原始数据里的最值，可以注释，不要删
    low_time = "2008-12-21 14:46:48 "
    high_time = "2009-01-01 00:00:00"

    timeLC = [low_time, high_time]

    a = _DataProcessor(userLC, problemLC, timeLC, 'kdd', TmpDir = "./DataProcessor/data")
    LCDataDir = a.LCDataDir
    saveDir = os.path.join(LCDataDir, 'KTM')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    batch_size=32 
    train_fraction=0.8
    dataset_params ={'active': active, 'window_lengths': window_lengths, 
                        'batch_size': batch_size, 'train_fraction': train_fraction}
    train_dataset, test_dataset = a.loadKTMData(dataset_params)

    feature_num = [d for d, l in train_dataset.take(1)][0].shape[-1]
    print('feature_num: ', feature_num)
    model_params = {} 
    model_params["feature_num"] = feature_num
    model_params["embed_dim"] = 20
    model_params["threshold"] = 0.5
    model_params["metrics_path"] = saveDir + "/metrics.csv"
    model = KTM(model_params)
    train(epoch=30, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
def runAssist():
    Features = {}
    Features['users'] = True
    Features['items'] = True
    Features['skills'] = True
    Features['lasttime_0kcsingle'] = False
    Features['lasttime_1kc'] = False
    Features['lasttime_2items'] = False
    Features['lasttime_3sequence'] = False
    Features['wins_1kc'] = False 
    Features['wins_2items'] = False
    Features['wins_3das3h'] = False #用于das3h中特征
    Features['wins_4das3hkc'] = False #用于das3h中特征
    Features['wins_5das3hitems'] = False #用于das3h中特征   
    Features['fails_1kc'] = False
    Features['attempts_1kc'] = False
    Features['attempts_2items'] = False
    Features['attempts_3das3h'] = False #用于das3h中特征
    Features['attempts_4das3hkc'] = False #用于das3h中特征
    Features['attempts_5das3hitems'] = False #用于das3h中特征
    active = [key for key, value in Features.items() if value]
    features_suffix = getFeaturesSuffix(active)
    window_lengths = [365 * 24 * 3600]
    ####################################### 
    # LC parameters
    #######################################
    userLC = [10,3000]
    problemLC = [10,3000]
    # algebra08原始数据里的最值，可以注释，不要删
    low_time = "2012-09-01 14:46:48 "
    high_time = "2012-09-30 00:00:00"

    timeLC = [low_time, high_time]

    a = _DataProcessor(userLC, problemLC, timeLC, 'assist', TmpDir = "./DataProcessor/data")
    LCDataDir = a.LCDataDir
    saveDir = os.path.join(LCDataDir, 'KTM')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    batch_size=32 
    train_fraction=0.8
    dataset_params ={'active': active, 'window_lengths': window_lengths, 
                        'batch_size': batch_size, 'train_fraction': train_fraction}
    train_dataset, test_dataset = a.loadKTMData(dataset_params)

    feature_num = [d for d, l in train_dataset.take(1)][0].shape[-1]
    print('feature_num: ', feature_num)
    model_params = {} 
    model_params["feature_num"] = feature_num
    model_params["embed_dim"] = 20
    model_params["threshold"] = 0.5
    model_params["metrics_path"] = saveDir + "/metrics.csv"
    model = KTM(model_params)
    train(epoch=30, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
if __name__ == "__main__":
    runKDD()
    # runAssist()