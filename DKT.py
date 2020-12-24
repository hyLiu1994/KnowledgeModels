#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import sys
import pickle
import tensorflow as tf 
from tensorflow.keras import layers

sys.path.append("./DataProcessor/")
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from public import *
from DataProcessor import _DataProcessor


class DKT(tf.keras.Model):
    def __init__(self, model_params):
        super(DKT, self).__init__()
        self.problem_num = model_params['problem_num'] 
        self.data_shape =  model_params['data_shape']
        self.lstm_units = model_params['lstm_units']
        self.problem_embed_dim = model_params['problem_embed_dim']
        self.dropout = model_params['dropout']
        self.l2 = model_params['l2']
        self.threshold = model_params['threshold']
        self.metrics_path = 'file://' + model_params['metrics_path']
        self.problem_embedding = layers.Embedding(input_dim=(self.problem_num) * 2 + 1, output_dim=self.problem_embed_dim, mask_zero=True)
        self.lstm = layers.LSTM(self.lstm_units, return_sequences=True, dropout=self.dropout, recurrent_regularizer=tf.keras.regularizers.l2(self.l2), bias_regularizer=tf.keras.regularizers.l2(self.l2))
        self.dense = layers.Dense(units=self.problem_num, kernel_regularizer=tf.keras.regularizers.l2(self.l2), bias_regularizer=tf.keras.regularizers.l2(self.l2), activation=tf.keras.activations.sigmoid)

        # train
        self.opti = tf.keras.optimizers.Adam()
        self.loss_obj = tf.keras.losses.BinaryCrossentropy()

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

    def call(self, inputs):
        
        problem_embed = self.problem_embedding(inputs)
        hidden = self.lstm(problem_embed)
        pred = self.dense(hidden)
        return pred

    def loss_function(self, pred, label):
        mask = pred._keras_mask
        pid, label = tf.split(label, num_or_size_splits=[-1, 1], axis=-1)
        pid = tf.cast(pid, dtype=tf.int32)
        # metrics
        pid = tf.one_hot(pid, depth=self.problem_num)
        pred = tf.expand_dims(pred, axis=-1)
        pred = tf.matmul(pid, pred)
        pred = tf.squeeze(pred, axis=-1)
        float_mask = tf.cast(mask, dtype=tf.float32)
        loss = self.loss_obj(label, pred, float_mask)
        # metrics
        self.metrics_loss.update_state(label, pred, mask)
        self.metrics_acc.update_state(label, pred, mask)
        self.metrics_pre.update_state(label, pred, mask)
        self.metrics_rec.update_state(label, pred, mask)
        self.metrics_auc.update_state(label, pred, mask)
        self.metrics_mae.update_state(label, pred, mask)
        self.metrics_rmse.update_state(label, pred, mask)
        return loss

    @tf.function(experimental_relax_shapes=True)
    def metrics_step(self, data, label):
        pred = self(data)
        self.loss_function(pred, label)
   
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data, label):
        with tf.GradientTape() as tape:
            pred = self(data)
            loss = self.loss_function(pred, label)
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
        return 

    def save_trainable_weights(self, save_path="./dkt.model"):
        data = tf.zeros(shape=self.data_shape)
        self(data)
        saved_variable = [w.numpy() for w in self.trainable_variables]
        
        fw = open(save_path, "wb")
        pickle.dump(saved_variable, fw)
        fw.close()
        print("save model to ", save_path)

    def load_trainable_weights(self, save_path="./dkt.model"):
        data = tf.zeros(shape=self.data_shape)
        self(data)
        if not os.path.exists(save_path):
            print("model parameters does not exists！")
            return
        fr = open(save_path, 'rb')
        saved_variable = pickle.load(fr)
        fr.close()
        for w, value in zip(self.trainable_variables, saved_variable):
            w.assign(value)
        print("load model from", save_path)

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
                    "train_loss: ", train_loss, 
                    "train_acc: ", train_acc, 
                    "train_pre: ", train_pre,
                    "train_rec: ", train_rec,
                    "train_auc: ", train_auc,
                    "train_mae: ", train_mae,
                    "train_rmse: ", train_rmse,
                    "test_loss: ", test_loss, 
                    "test_acc: ", test_acc, 
                    "test_pre: ", test_pre,
                    "test_rec: ", test_rec,
                    "test_auc: ", test_auc,
                    "test_mae: ", test_mae,
                    "test_rmse: ", test_rmse)
            start = tf.timestamp()

def get_last_epoch_data(model, dataset):
    all_pred = list()
    all_label = list()
    model.resetMetrics()
    for data, label in dataset:
        pred = model.metrics_step(data, label)
    return (model.metrics_acc.result(), model.metrics_pre.result(), model.metrics_rec.result(), 
    model.metrics_auc.result(), model.metrics_mae.result(), model.metrics_rmse.result())

def runKDD(is_test=True):
    #######################################
    # LC parameters
    #######################################
    userLC = [10,3000]
    problemLC = [10,5000]
    # algebra08原始数据里的最值，可以注释，不要删
    low_time = "2008-12-21 14:46:48"
    high_time = "2009-01-01 00:00:00"
    timeLC = [low_time, high_time]
    data_processor = _DataProcessor(userLC, problemLC, timeLC, 'kdd', TmpDir = "./DataProcessor/data")

    LCDataDir = data_processor.LCDataDir
    saveDir = os.path.join(LCDataDir, 'DKT')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    LC_params = data_processor.LC_params

    dataset_params = copy.deepcopy(LC_params)
    dataset_params["trainRate"] = 0.8
    dataset_params["batch_size"] = 32

    [train_dataset, test_dataset, problem_num] = data_processor.loadDKTbatchData(dataset_params)
    #######################################
    # model parameters
    #######################################
    model_params = {}
    model_params['trainRate'] = 0.8

    model_params['lstm_units'] = 40
    model_params['dropout'] = 0.01
    model_params['l2'] = 0.01
    model_params['problem_embed_dim'] = 20
    model_params['problem_num'] = problem_num
    model_params['epoch'] = 200
    model_params['threshold'] = 0.5
    model_params['metrics_path'] = saveDir + '/metrics.csv'
    model_params["data_shape"] = [data for data, label in train_dataset.take(1)][0].shape.as_list()

    model = DKT(model_params)
    if is_test:
        train_dataset = train_dataset.take(10)
        test_dataset = test_dataset.take(8)

    #######################################
    # train parameters
    #######################################
    train(epoch=model_params['epoch'], model=model, train_dataset=train_dataset, test_dataset=test_dataset)

    #######################################
    # save model
    #######################################
    results={'LC_params':LC_params, 'model_params':model_params,'results':{}}
    temp = results['results']
    [temp['tf_Accuracy'], temp['tf_Precision'], temp['tf_Recall'], temp['tf_AUC'], temp['tf_MAE'], temp['tf_RMSE']] = get_last_epoch_data(model, test_dataset)

    model_params.pop("metrics_path")
    saveDict(results, saveDir, 'results'+ getLegend(model_params)+'.json')


def runOJ(is_test=True):
    #######################################
    # LC parameters
    #######################################
    userLC = [10,500,0.1,1]
    problemLC = [10,500,0,1]
    #hdu原始数据里的最值，可以注释，不要删
    low_time = "2018-11-22 00:00:00" 
    high_time = "2018-11-29 00:00:00"
    timeLC = [low_time, high_time]
    data_processor = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir = "./DataProcessor/data")

    LCDataDir = data_processor.LCDataDir
    saveDir = os.path.join(LCDataDir, 'DKT')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    LC_params = data_processor.LC_params

    dataset_params = copy.deepcopy(LC_params)
    dataset_params["trainRate"] = 0.8
    dataset_params["batch_size"] = 32

    [train_dataset, test_dataset, problem_num] = data_processor.loadDKTbatchData(dataset_params)
    #######################################
    # model parameters
    #######################################
    model_params = {}
    model_params['trainRate'] = 0.8

    model_params['lstm_units'] = 40
    model_params['dropout'] = 0.01
    model_params['l2'] = 0.01
    model_params['problem_embed_dim'] = 20
    model_params['problem_num'] = problem_num
    model_params['epoch'] = 200
    model_params['threshold'] = 0.5
    model_params['metrics_path'] = saveDir + '/metrics.csv'
    model_params["data_shape"] = [data for data, label in train_dataset.take(1)][0].shape.as_list()

    model = DKT(model_params)
    if is_test:
        train_dataset = train_dataset.take(10)
        test_dataset = test_dataset.take(8)

    #######################################
    # train parameters
    #######################################
    train(epoch=model_params['epoch'], model=model, train_dataset=train_dataset, test_dataset=test_dataset)

    #######################################
    # save model
    #######################################
    results={'LC_params':LC_params, 'model_params':model_params,'results':{}}
    temp = results['results']
    [temp['tf_Accuracy'], temp['tf_Precision'], temp['tf_Recall'], temp['tf_AUC'], temp['tf_MAE'], temp['tf_RMSE']] = get_last_epoch_data(model, test_dataset)

    model_params.pop("metrics_path")
    saveDict(results, saveDir, 'results'+ getLegend(model_params)+'.json')


def runAssist(is_test=True):
    #######################################
    # LC parameters
    #######################################
    userLC = [10,3000]
    problemLC = [10,3000]
    #hdu原始数据里的最值，可以注释，不要删
    low_time = "2012-09-01 00:00:00" 
    high_time = "2012-09-30 00:00:00"
    timeLC = [low_time, high_time]
    data_processor = _DataProcessor(userLC, problemLC, timeLC, 'assist', TmpDir = "./DataProcessor/data")


    LCDataDir = data_processor.LCDataDir
    saveDir = os.path.join(LCDataDir, 'DKT')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    LC_params = data_processor.LC_params

    dataset_params = copy.deepcopy(LC_params)
    dataset_params["trainRate"] = 0.8
    dataset_params["batch_size"] = 32

    [train_dataset, test_dataset, problem_num] = data_processor.loadDKTbatchData(dataset_params)
    #######################################
    # model parameters
    #######################################
    model_params = {}
    model_params['trainRate'] = 0.8

    model_params['lstm_units'] = 40
    model_params['dropout'] = 0.01
    model_params['l2'] = 0.01
    model_params['problem_embed_dim'] = 20
    model_params['problem_num'] = problem_num
    model_params['epoch'] = 200
    model_params['threshold'] = 0.5
    model_params['metrics_path'] = saveDir + '/metrics.csv'
    model_params["data_shape"] = [data for data, label in train_dataset.take(1)][0].shape.as_list()

    model = DKT(model_params)
    if is_test:
        train_dataset = train_dataset.take(10)
        test_dataset = test_dataset.take(8)

    #######################################
    # train parameters
    #######################################
    train(epoch=model_params['epoch'], model=model, train_dataset=train_dataset, test_dataset=test_dataset)

    #######################################
    # save model
    #######################################
    results={'LC_params':LC_params, 'model_params':model_params,'results':{}}
    temp = results['results']
    [temp['tf_Accuracy'], temp['tf_Precision'], temp['tf_Recall'], temp['tf_AUC'], temp['tf_MAE'], temp['tf_RMSE']] = get_last_epoch_data(model, test_dataset)

    model_params.pop("metrics_path")
    saveDict(results, saveDir, 'results'+ getLegend(model_params)+'.json')
    
def set_run_eagerly(is_eager=False):
    if tf.__version__ == "2.2.0":
        tf.config.experimental_run_functions_eagerly(is_eager)
    else:
        tf.config.run_functions_eagerly(is_eager)
if __name__ == "__main__":
    set_run_eagerly(False)
    runOJ(False)
    runKDD(False)
    runAssist(False)


