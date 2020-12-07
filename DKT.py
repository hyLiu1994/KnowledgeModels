#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("./DataProcessor/")
import tensorflow as tf
import os
import pickle
from tensorflow.keras import layers
from DataProcessor import _DataProcessor
from public import *

class DKT(tf.keras.Model):
    def __init__(self, data_shape, lstm_units, dropout, l2, problem_embed_dim, problem_num, **kwargs):
        super(DKT, self).__init__(**kwargs)
        self.problem_num = problem_num
        self.data_shape = data_shape
        self.problem_embedding = layers.Embedding(input_dim=(problem_num) * 2 + 1, output_dim=problem_embed_dim, mask_zero=True)
        self.lstm = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout, recurrent_regularizer=tf.keras.regularizers.l2(l2), bias_regularizer=tf.keras.regularizers.l2(l2))
        self.dense = layers.Dense(units=problem_num, kernel_regularizer=tf.keras.regularizers.l2(l2), bias_regularizer=tf.keras.regularizers.l2(l2), activation=tf.keras.activations.sigmoid)

        # train
        self.opti = tf.keras.optimizers.Adam()
        self.loss_obj = tf.keras.losses.BinaryCrossentropy()

        # metrics
        self.metrics_loss = tf.keras.metrics.BinaryCrossentropy()
        self.metrics_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.metrics_auc = tf.keras.metrics.AUC()

    def call(self, inputs):
        problem_embed = self.problem_embedding(inputs)
        print (problem_embed.shape)
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
        self.metrics_accuracy.update_state(label, pred, mask)
        self.metrics_loss.update_state(label, pred, mask)
        self.metrics_auc.update_state(label, pred, mask)
        return loss

    @tf.function(experimental_relax_shapes=True)
    def metrics_step(self, data, label):
        pred = self(data)
        self.loss_function(pred, label)
   
    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, data, label):
        with tf.GradientTape() as tape:
            pred = self(data)
            loss = self.loss_function(pred, label)
        grad = tape.gradient(loss, self.trainable_variables)
        self.opti.apply_gradients(zip(grad, self.trainable_variables))

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
    tf.print("test loss: ", model.metrics_loss.result(), "acc: ", model.metrics_accuracy.result(), "auc: ", model.metrics_auc.result())
    model.metrics_loss.reset_states()
    model.metrics_accuracy.reset_states()
    model.metrics_auc.reset_states()

@tf.function
def train(epoch, model, train_dataset, test_dataset):
    element_num = tf.data.experimental.cardinality(train_dataset)
    for i, (data, label) in train_dataset.repeat(epoch).enumerate():
        model.train_step(data, label)
        if tf.equal(tf.math.floormod(i, element_num), 0):
            tf.print("epoch: ", tf.math.floordiv(i, element_num), "train loss: ", model.metrics_loss.result(), "acc: ", model.metrics_accuracy.result(), "auc: ", model.metrics_auc.result(), end=", ")
            model.metrics_loss.reset_states()
            model.metrics_accuracy.reset_states()
            model.metrics_auc.reset_states()
            test(model, test_dataset)

def get_last_epoch_data(model, dataset):
    all_pred = list()
    all_label = list()
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
    epoch = 1000
    threshold = 0.5

    model_params = {}
    model_params['trainRate'] = trainRate

    model_params['lstm_units'] = lstm_units
    model_params['dropout'] = dropout
    model_params['l2'] = l2
    model_params['problem_embed_dim'] = problem_embed_dim
    model_params['epoch'] = epoch
    model_params['threshold'] = threshold

    batch_size = 128

    #######################################
    # LC parameters
    #######################################
    userLC = [10,3000]
    problemLC = [10,5000]
    #algebra08原始数据里的最值，可以注释，不要删
    low_time = "2008-09-08 14:46:48"
    high_time = "2009-01-01 00:00:00"
    timeLC = [low_time, high_time]
    a = _DataProcessor(userLC, problemLC, timeLC, 'kdd', TmpDir = "./DataProcessor/data/")

    LCDataDir = a.LCDataDir
    saveDir = os.path.join(LCDataDir, 'dkt')
    prepareFolder(saveDir)
    LC_params = a.LC_params

    [train_dataset, test_dataset, problem_num] = a.loadDKTbatchData(trainRate, batch_size)
    print (lstm_units, dropout, l2, problem_embed_dim, epoch, problem_num, threshold)
    model = DKT(lstm_units, dropout, l2, problem_embed_dim, epoch, problem_num, threshold)

    is_test = True
    if is_test:
        train_dataset = train_dataset.take(10)
        test_dataset = test_dataset.take(8)

    train(epoch=epoch, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
    results={'LC_params':LC_params,'model_params':model_params,'results':{}}
    temp = results['results']
    [temp['tf_Accuracy'],temp['tf_Precision'],temp['tf_Recall'],temp['tf_MSE'],temp['tf_MAE'],temp['tf_RMSE'],temp['tf_AUC'],temp['tf_AUC_1000']] = get_last_epoch_data(model, test_dataset)
    saveDict(results, saveDir, 'results'+getLegend(model_params)+'.json')

    
if __name__ == "__main__":
    runKDD()


