#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import pickle
import sys
# from data_processor_Assistment_2015 import Assistments2015_data_processor
sys.path.append("./DataProcessor/")
from DataProcessor import _DataProcessor
from public import *

class DKVMN(tf.keras.Model):

    def __init__(self, model_params):
        super(DKVMN, self).__init__()



        self.data_shape = model_params['data_shape']
        self.problem_num = model_params['problem_num']
        self.m_N = model_params['m_N']
        self.mk_dim = model_params['mk_dim']
        self.mv_dim = model_params['mv_dim']
        self.threshold = model_params['threshold']
    
        self.metrics_path= 'file://' + model_params['metrics_path']
        
        self.problem_embed = tf.keras.layers.Embedding(input_dim=self.problem_num+1, output_dim=self.mk_dim)
        self.problem_label_embed = tf.keras.layers.Embedding(input_dim=2*self.problem_num+1, output_dim=self.mv_dim, mask_zero=False)
        self.mk = self.add_weight(name="mk", shape=(self.m_N, self.mk_dim))
        self.mv = self.add_weight(name="mv", shape=(self.m_N, self.mk_dim))

        self.dense_f = tf.keras.layers.Dense(units=50, activation="tanh")
        self.dense_p = tf.keras.layers.Dense(units=1, activation="sigmoid")

        self.dense_E = tf.keras.layers.Dense(units=self.mv_dim, activation="sigmoid")
        self.dense_A = tf.keras.layers.Dense(units=self.mv_dim, activation="tanh")
        
        # loss
        self.loss = tf.keras.losses.BinaryCrossentropy()
        # opti
        self.opti = tf.keras.optimizers.Adam()
        
        # metrics
        self.metrics_format = "epoch,time,train_loss,train_acc,train_pre,train_rec,train_auc,train_mae,train_rmse,test_loss,test_acc,test_pre,test_rec,test_auc,test_mae,test_rmse"
        self.metrics_loss = tf.keras.metrics.BinaryCrossentropy()

        self.metrics_acc = tf.keras.metrics.BinaryAccuracy(threshold = self.threshold)
        self.metrics_pre = tf.keras.metrics.Precision(thresholds = self.threshold)
        self.metrics_rec = tf.keras.metrics.Recall(thresholds = self.threshold)
        self.metrics_auc = tf.keras.metrics.AUC()
        self.metrics_mae = tf.keras.metrics.MeanAbsoluteError()
        self.metrics_rmse = tf.keras.metrics.RootMeanSquaredError()

    def resetMetrics(self):
        self.metrics_loss.reset_states()
        self.metrics_acc.reset_states()
        self.metrics_pre.reset_states()
        self.metrics_rec.reset_states()
        self.metrics_auc.reset_states()
        self.metrics_mae.reset_states()
        self.metrics_rmse.reset_states()
        return 

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
        state = tf.tile(tf.expand_dims(self.mv, axis=0), [tf.shape(inputs)[0], 1, 1])
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
        self.metrics_acc.update_state(label, pred, mask)
        self.metrics_pre.update_state(label, pred, mask)
        self.metrics_rec.update_state(label, pred, mask)
        self.metrics_auc.update_state(label, pred, mask)
        self.metrics_mae.update_state(label, pred, mask)
        self.metrics_rmse.update_state(label, pred, mask)
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
    return (model.metrics_acc.result(), model.metrics_pre.result(), model.metrics_rec.result(), 
    model.metrics_auc.result(), model.metrics_mae.result(), model.metrics_rmse.result())

@tf.function
def test(model, test_dataset):
    for data, label in test_dataset:
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
    saveDir = os.path.join(LCDataDir, 'DKVMN')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    LC_params = data_processor.LC_params

    dataset_params = copy.deepcopy(LC_params)
    dataset_params["trainRate"] = 0.8
    dataset_params["batch_size"] = 32

    [train_dataset, test_dataset, problem_num] = data_processor.loadDKVMNbatchData(dataset_params)
    #######################################
    # model parameters
    #######################################
    model_params = {}
    model_params['m_N'] = 40
    model_params['mk_dim'] = 60
    model_params['mv_dim'] = 60
    model_params['threshold'] = 0.5
    model_params['data_shape'] = [data for data, label in train_dataset][0].shape.as_list()
    model_params['problem_num'] = problem_num
    model_params['epoch'] = 2
    model_params['metrics_path'] = saveDir + '/metrics.csv'

    model = DKVMN(model_params)
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
    saveDir = os.path.join(LCDataDir, 'DKVMN')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    LC_params = data_processor.LC_params

    dataset_params = copy.deepcopy(LC_params)
    dataset_params["trainRate"] = 0.8
    dataset_params["batch_size"] = 32

    [train_dataset, test_dataset, problem_num] = data_processor.loadDKVMNbatchData(dataset_params)
    #######################################
    # model parameters
    #######################################
    model_params = {}
    model_params['m_N'] = 40
    model_params['mk_dim'] = 60
    model_params['mv_dim'] = 60
    model_params['threshold'] = 0.5
    model_params['data_shape'] = [data for data, label in train_dataset][0].shape.as_list()
    model_params['problem_num'] = problem_num
    model_params['epoch'] = 2
    model_params['metrics_path'] = saveDir + '/metrics.csv'

    model = DKVMN(model_params)
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
    runKDD(False)
    # runOJ(True)


