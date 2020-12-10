#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import pickle
import sys
# from data_processor_Assistment_2015 import Assistments2015_data_processor
sys.path.append("./DataProcessor/")
from DataProcessor import _DataProcessor
from public import *

class DKVMN(tf.keras.Model):

    def __init__(self, data_shape, problem_num, m_N, mk_dim, mv_dim, threshold):
        super(DKVMN, self).__init__()

        self.data_shape = data_shape
        self.problem_num = problem_num
        self.m_N = m_N
        self.mk_dim = mk_dim
        self.mv_dim = mv_dim
        
        self.problem_embed = tf.keras.layers.Embedding(input_dim=problem_num+1, output_dim=mk_dim)
        self.problem_label_embed = tf.keras.layers.Embedding(input_dim=2*problem_num+1, output_dim=mv_dim, mask_zero=False)
        self.mk = self.add_weight(name="mk", shape=(self.m_N, self.mk_dim))

        self.dense_f = tf.keras.layers.Dense(units=50, activation="tanh")
        self.dense_p = tf.keras.layers.Dense(units=1, activation="sigmoid")

        self.dense_E = tf.keras.layers.Dense(units=self.mv_dim, activation="sigmoid")
        self.dense_A = tf.keras.layers.Dense(units=self.mv_dim, activation="tanh")
        
        # loss
        self.loss = tf.keras.losses.BinaryCrossentropy()
        # opti
        self.opti = tf.keras.optimizers.Adam()
        
        # metrics
        self.metrics_loss = tf.keras.metrics.BinaryCrossentropy()
        self.metrics_accuracy = tf.keras.metrics.BinaryAccuracy(threshold = threshold)
        self.metrics_precision = tf.keras.metrics.Precision(thresholds = threshold)
        self.metrics_recall = tf.keras.metrics.Recall(thresholds = threshold)
        self.metrics_mse = tf.keras.metrics.MeanSquaredError()
        self.metrics_mae = tf.keras.metrics.MeanAbsoluteError()
        self.metrics_rmse = tf.keras.metrics.RootMeanSquaredError()
        self.metrics_auc = tf.keras.metrics.AUC()
        self.metrics_auc_1000 = tf.keras.metrics.AUC(num_thresholds=1000)


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
    def get_initial_state(self, inputs):
        init_mv = tf.zeros(shape=(tf.shape(inputs)[0], self.m_N, self.mv_dim), dtype=tf.float32)
        return init_mv

    def get_p_t(self, k_t, w_t, mv_tm1):
        r_t = tf.matmul(w_t, mv_tm1)
        r_t_k_t = tf.concat([r_t, k_t], axis=-1)
        f_t = self.dense_f(r_t_k_t)
        p_t = self.dense_p(f_t)
        p_t = tf.squeeze(p_t, axis=-1)
        return p_t

    def get_mv_t(self, w_t, skill_with_correct, mv_tm1):
        v_t = self.problem_label_embed(skill_with_correct)
        # e_t [batch_size, 1, mv_dim]
        # w_t [batch_size, 1, m_N]
        e_t = self.dense_E(v_t)
        mv_e = tf.matmul(w_t, e_t, transpose_a=True)
        mv_e = tf.ones_like(mv_e) - mv_e
        mv_t = tf.multiply(mv_e, mv_tm1)
        a_t = self.dense_A(v_t)
        mv_t = mv_t + tf.matmul(w_t, a_t, transpose_a=True)
        return mv_t

    def save_trainable_weights(self, save_path="./dkvmn.model"):
        data = tf.zeros(shape=self.data_shape)
        self(data)
        saved_variable = [w.numpy() for w in self.trainable_variables]
        
        fw = open(save_path, "wb")
        pickle.dump(saved_variable, fw)
        fw.close()
        print("save model to ", save_path)

    def load_trainable_weights(self, save_path="./dkvmn.model"):
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

    def time_step(self, inputs, state_tm1):
        # inputs: [batch_size, feature_dim]
        mv_tm1 = state_tm1
        # ori_skill: [batch_size, 1] skill_with_correct: [batch_size, 1]
        ori_skill, skill_with_correct = tf.split(inputs,  num_or_size_splits=2, axis=-1)
        # k_t: [batch_size, 1, mk_dimp]
        k_t = self.problem_embed(ori_skill)
        # mk: [N, mk_dim]
        w_t = tf.keras.activations.softmax(tf.matmul(k_t, self.mk, transpose_b=True))
        # p_t: [batch_size, 1]
        p_t = self.get_p_t(k_t, w_t, mv_tm1)
        # mv_t: [batch_size, m_N, mv_dim]
        mv_t = self.get_mv_t(w_t, skill_with_correct, mv_tm1)
        return p_t, mv_t

    def call(self, inputs):
        time_inputs = tf.transpose(inputs, perm=[1, 0, 2])
        state = self.get_initial_state(inputs)
        pred = tf.TensorArray(dtype=tf.float32, size=tf.shape(time_inputs)[0])
        for i in tf.range(tf.shape(time_inputs)[0]):
            p_t, state = self.time_step(time_inputs[i], state)
            pred = pred.write(i, p_t)
        pred = pred.stack()
        pred = tf.transpose(pred, perm=[1, 0, 2])
        return pred

    def loss_function(self, pred, label, mask):
        loss = self.loss(label, pred, mask)
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

    def get_mask(self, data):
        ori_skill, skill_with_correct = tf.split(data, num_or_size_splits=2, axis=-1)
        mask = tf.cast(tf.not_equal(ori_skill, 0), dtype=tf.float32)
        mask = tf.squeeze(mask, axis=-1)
        return mask
    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data, label):
        mask = self.get_mask(data)
        with tf.GradientTape() as tape:
            pred = self(data)
            loss = self.loss_function(pred, label, mask)
        grad = tape.gradient(loss, self.trainable_variables)
        self.opti.apply_gradients(zip(grad, self.trainable_variables))

    @tf.function(experimental_relax_shapes=True)
    def metrics_step(self, data, label):
        mask = self.get_mask(data)
        pred = self(data)
        self.loss_function(pred, label, mask)

def get_last_epoch_data(model, dataset):
    all_pred = list()
    all_label = list()
    model.resetMetrics()
    for data, label in dataset:
        pred = model.metrics_step(data, label)
    return model.metrics_accuracy.result(),model.metrics_precision.result(),model.metrics_recall.result(),model.metrics_mse.result(),model.metrics_mae.result(), model.metrics_rmse.result(), model.metrics_auc.result(), model.metrics_auc_1000.result()
@tf.function
def test(model, test_dataset):
    for data, label in test_dataset:
        model.metrics_step(data, label)
    tf.print("test dataset loss: ", model.metrics_loss.result(), "acc: ", model.metrics_accuracy.result(), "auc: ", model.metrics_auc.result())
    model.resetMetrics()

@tf.function
def train(epoch, model, train_dataset, test_dataset):
    element_num = tf.data.experimental.cardinality(train_dataset)
    start = tf.timestamp()
    for i, (data, label) in train_dataset.repeat(epoch).enumerate():
        model.train_step(data, label)
        if tf.equal(tf.math.floormod(i, element_num), 0):
            end = tf.timestamp()
            tf.print("epoch: ", tf.math.floordiv(i, element_num),
                     "train loss: ", model.metrics_loss.result(),
                     "acc: ", model.metrics_accuracy.result(),
                     "auc: ", model.metrics_auc.result(),
                     "time: ", end - start, end=",")
            model.resetMetrics()
            test(model, test_dataset)

            start = tf.timestamp()


def runOJ():
    #######################################
    # model parameters
    #######################################
    trainRate = 0.8
    lstm_units = 40
    dropout = 0.01
    l2 = 0.01
    problem_embed_dim = 20
    epoch = 4
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
    userLC = [10,500,0.1,1]
    problemLC = [10,500,0,1]
    #hdu原始数据里的最值，可以注释，不要删
    low_time = "2018-11-22 00:00:00" 
    high_time = "2018-11-29 00:00:00"
    timeLC = [low_time, high_time]
    a = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir = "data")

    LCDataDir = a.LCDataDir
    saveDir = os.path.join(LCDataDir, 'dkvmn')
    prepareFolder(saveDir)
    LC_params = a.LC_params

    [train_dataset, test_dataset, problem_num] = a.loadDKVMNbatchData(trainRate, batch_size)
    m_N = 40
    mk_dim = 60
    mv_dim = 60
    data_shape = [data for data, label in train_dataset][0].shape
    model = DKVMN(data_shape=data_shape, problem_num=problem_num, m_N=m_N, mk_dim=mk_dim, mv_dim=mv_dim, threshold=threshold)
    is_test = False
    if is_test:
        train_dataset = train_dataset.take(10)
        test_dataset = test_dataset.take(8)

    train(epoch=epoch, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
    results={'LC_params':LC_params,'model_params':model_params,'results':{}}
    temp = results['results']
    [temp['tf_Accuracy'],temp['tf_Precision'],temp['tf_Recall'],temp['tf_MSE'],temp['tf_MAE'],temp['tf_RMSE'],temp['tf_AUC'],temp['tf_AUC_1000']] = get_last_epoch_data(model, test_dataset)
    saveDict(results, saveDir, 'results'+ getLegend(model_params)+'.json')

    
if __name__ == "__main__":
    tf.config.run_functions_eagerly(True)
    runOJ()


