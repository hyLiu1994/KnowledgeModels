#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pickle
import tensorflow as tf 
from tensorflow.keras import layers

sys.path.append("./DataProcessor/")
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from public import *
from DataProcessor import _DataProcessor


class DKT(tf.keras.Model):
    def __init__(self, data_shape, lstm_units, dropout, l2, problem_embed_dim, problem_num, threshold, **kwargs):
        super(DKT, self).__init__(**kwargs)
        self.problem_num = problem_num
        self.data_shape = data_shape
        self.problem_embedding = layers.Embedding(input_dim=(problem_num) * 2 + 1, output_dim=problem_embed_dim, mask_zero=True)
        self.lstm = layers.LSTM(lstm_units, return_sequences=True,
                                dropout=dropout, recurrent_regularizer=tf.keras.regularizers.l2(l2),
                                bias_regularizer=tf.keras.regularizers.l2(l2))
        self.dense = layers.Dense(units=problem_num,
                                  kernel_regularizer=tf.keras.regularizers.l2(l2),
                                  bias_regularizer=tf.keras.regularizers.l2(l2),
                                  activation=tf.keras.activations.sigmoid)
        # train
        self.opti = tf.keras.optimizers.Adam()
        self.loss_obj = tf.keras.losses.BinaryCrossentropy()

        # metrics
        self.metrics_loss = tf.keras.metrics.BinaryCrossentropy()
        self.metrics_accuracy = tf.keras.metrics.BinaryAccuracy(
            threshold=threshold)
        self.metrics_precision = tf.keras.metrics.Precision(
            thresholds=threshold)
        self.metrics_recall = tf.keras.metrics.Recall(thresholds=threshold)
        self.metrics_mse = tf.keras.metrics.MeanSquaredError()
        self.metrics_mae = tf.keras.metrics.MeanAbsoluteError()
        self.metrics_rmse = tf.keras.metrics.RootMeanSquaredError()
        self.metrics_auc = tf.keras.metrics.AUC()
        self.metrics_auc_1000 = tf.keras.metrics.AUC(num_thresholds=1000)

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
        self.metrics_accuracy.update_state(label, pred, mask)
        self.metrics_precision.update_state(label, pred, mask)
        self.metrics_recall.update_state(label, pred, mask)
        self.metrics_mse.update_state(label, pred, mask)
        self.metrics_mae.update_state(label, pred, mask)
        self.metrics_rmse.update_state(label, pred, mask)
        self.metrics_auc.update_state(label, pred, mask)
        self.metrics_auc_1000.update_state(label, pred, mask)
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
        self.metrics_accuracy.reset_states()
        self.metrics_precision.reset_states()
        self.metrics_recall.reset_states()
        self.metrics_mse.reset_states()
        self.metrics_mae.reset_states()
        self.metrics_rmse.reset_states()
        self.metrics_auc.reset_states()
        self.metrics_auc_1000.reset_states()
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
    tf.print("test loss: ", model.metrics_loss.result(),
             "acc: ", model.metrics_accuracy.result(),
             "auc: ", model.metrics_auc.result())
    model.resetMetrics()

@tf.function
def train(epoch, model, train_dataset, test_dataset):
    element_num = tf.data.experimental.cardinality(train_dataset)
    start = tf.timestamp()
    for i, (data, label) in train_dataset.repeat(epoch).enumerate():
        # tf.print(i, element_num, len(data), "acc: ", model.metrics_accuracy.result(), "auc: ", model.metrics_auc.result(),
        # "pre: ", model.metrics_precision.result(), "recall: ", model.metrics_recall.result(), "rmse: ", model.metrics_rmse.result(), 
        # "mae: ", model.metrics_mae.result(), "mse: ", model.metrics_mse.result())
        model.train_step(data, label)
        if tf.equal(tf.math.floormod(i, element_num), 0):
            end = tf.timestamp()
            tf.print("epoch: ", tf.math.floordiv(i, element_num), "train loss: ", model.metrics_loss.result(), 
            "acc: ", model.metrics_accuracy.result(), "auc: ", model.metrics_auc.result(), "costTime: ", end - start , end=", ")
            start = end
            model.resetMetrics()
            test(model, test_dataset)

def get_last_epoch_data(model, dataset):
    all_pred = list()
    all_label = list()
    model.resetMetrics()
    for data, label in dataset:
        pred = model.metrics_step(data, label)
    return model.metrics_accuracy.result(),model.metrics_precision.result(),model.metrics_recall.result(),model.metrics_mse.result(),model.metrics_mae.result(), model.metrics_rmse.result(), model.metrics_auc.result(), model.metrics_auc_1000.result()

def runKDD():
    #######################################
    # model parameters
    #######################################
    trainRate = 0.8
    lstm_units = 40
    dropout = 0.01
    l2 = 0.01
    problem_embed_dim = 20
    epoch = 1
    threshold = 0.5

    model_params = {}
    model_params['trainRate'] = trainRate

    model_params['lstm_units'] = lstm_units
    model_params['dropout'] = dropout
    model_params['l2'] = l2
    model_params['problem_embed_dim'] = problem_embed_dim
    model_params['epoch'] = epoch
    model_params['threshold'] = threshold

    batch_size = 32

    #######################################
    # LC parameters
    #######################################
    userLC = [10,3000]
    problemLC = [10,5000]
    # algebra08原始数据里的最值，可以注释，不要删
    low_time = "2008-09-08 14:46:48"
    high_time = "2009-01-01 00:00:00"
    timeLC = [low_time, high_time]
    a = _DataProcessor(userLC, problemLC, timeLC, 'kdd', TmpDir = "./data")

    LCDataDir = a.LCDataDir
    saveDir = os.path.join(LCDataDir, 'dkt')
    prepareFolder(saveDir)
    LC_params = a.LC_params

    [train_dataset, test_dataset, problem_num] = a.loadDKTbatchData(trainRate, batch_size)

    data_shape = [data for data, label in train_dataset.take(1)][0].shape
    model = DKT(data_shape, lstm_units, dropout, l2, problem_embed_dim, problem_num, threshold)

    is_test = False
    if is_test:
        print ("测试运行DKT")
        train_dataset = train_dataset.take(1)
        test_dataset = test_dataset.take(1)

    train(epoch=epoch, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
    model.save_trainable_weights(saveDir + "/dkt.model")
    results={'LC_params':LC_params,'model_params':model_params,'results':{}}
    temp = results['results']
    [temp['tf_Accuracy'],temp['tf_Precision'],temp['tf_Recall'],temp['tf_MSE'],temp['tf_MAE'],temp['tf_RMSE'],temp['tf_AUC'],temp['tf_AUC_1000']] = get_last_epoch_data(model, test_dataset)
    saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')

def runOJ():
    #######################################
    # model parameters
    #######################################
    trainRate = 0.8
    lstm_units = 40
    dropout = 0.01
    l2 = 0.01
    problem_embed_dim = 20
    epoch = 5000
    threshold = 0.5

    model_params = {}
    model_params['trainRate'] = trainRate

    model_params['lstm_units'] = lstm_units
    model_params['dropout'] = dropout
    model_params['l2'] = l2
    model_params['problem_embed_dim'] = problem_embed_dim
    model_params['epoch'] = epoch
    model_params['threshold'] = threshold

    batch_size = 32

    #######################################
    # LC parameters
    #######################################
    userLC = [10, 500, 0.1, 1]
    problemLC = [10, 500, 0, 1]
    # hdu原始数据里的最值，可以注释，不要删
    low_time = "2018-06-01 00:00:00"
    low_time = "2018-11-22 00:00:00"
    high_time = "2018-11-29 00:00:00"
    timeLC = [low_time, high_time]
    a = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir="./DataProcessor/data/")

    LCDataDir = a.LCDataDir
    saveDir = os.path.join(LCDataDir, 'dkt')
    prepareFolder(saveDir)
    LC_params = a.LC_params

    [train_dataset, test_dataset, problem_num] = a.loadDKTbatchData(trainRate, batch_size)

    data_shape = [data for data, label in train_dataset.take(1)][0].shape
    model = DKT(data_shape, lstm_units, dropout, l2, problem_embed_dim, problem_num, threshold)

    is_test = False
    if is_test:
        print ("测试运行DKT")
        train_dataset = train_dataset.take(1)
        test_dataset = test_dataset.take(1)

    train(epoch=epoch, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
    model.save_trainable_weights(saveDir + "/dkt.model")
    results={'LC_params':LC_params,'model_params':model_params,'results':{}}
    temp = results['results']
    [temp['tf_Accuracy'],temp['tf_Precision'],temp['tf_Recall'],temp['tf_MSE'],temp['tf_MAE'],temp['tf_RMSE'],temp['tf_AUC'],temp['tf_AUC_1000']] = get_last_epoch_data(model, test_dataset)
    saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')


def runAssist():
    #######################################
    # model parameters
    #######################################
    trainRate = 0.8
    lstm_units = 40
    dropout = 0.01
    l2 = 0.01
    problem_embed_dim = 20
    epoch = 1
    threshold = 0.5

    model_params = {}
    model_params['trainRate'] = trainRate

    model_params['lstm_units'] = lstm_units
    model_params['dropout'] = dropout
    model_params['l2'] = l2
    model_params['problem_embed_dim'] = problem_embed_dim
    model_params['epoch'] = epoch
    model_params['threshold'] = threshold

    batch_size = 32

    #######################################
    # LC parameters
    #######################################
    userLC = [10,20]
    problemLC = [10,20]
    #assistments12原始数据里的最值，可以注释，不要删
    low_time = "2012-09-01 00:00:00"
    high_time = "2013-09-01 00:00:00"
    timeLC = [low_time, high_time]
    a = _DataProcessor(userLC, problemLC, timeLC, 'assist', TmpDir = "./data")

    LCDataDir = a.LCDataDir
    saveDir = os.path.join(LCDataDir, 'dkt')
    prepareFolder(saveDir)
    LC_params = a.LC_params

    [train_dataset, test_dataset, problem_num] = a.loadDKTbatchData(trainRate, batch_size)

    data_shape = [data for data, label in train_dataset.take(1)][0].shape
    model = DKT(data_shape, lstm_units, dropout, l2, problem_embed_dim, problem_num, threshold)

    is_test = False
    if is_test:
        print ("测试运行DKT")
        train_dataset = train_dataset.take(1)
        test_dataset = test_dataset.take(1)

    train(epoch=epoch, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
    model.save_trainable_weights(saveDir + "/dkt.model")
    results={'LC_params':LC_params,'model_params':model_params,'results':{}}
    temp = results['results']
    [temp['tf_Accuracy'],temp['tf_Precision'],temp['tf_Recall'],temp['tf_MSE'],temp['tf_MAE'],temp['tf_RMSE'],temp['tf_AUC'],temp['tf_AUC_1000']] = get_last_epoch_data(model, test_dataset)
    saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')



    
if __name__ == "__main__":
    runOJ()
    # runAssist()


