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

def load_data(batch_size=32, train_fraction=0.8):
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
    # LC parameters
    #######################################
    userLC = [10,100]
    problemLC = [10,100]
    # algebra08原始数据里的最值，可以注释，不要删
    low_time = "2008-09-08 14:46:48"
    high_time = "2009-01-01 00:00:00"

    timeLC = [low_time, high_time]
    a = _DataProcessor(userLC, problemLC, timeLC, 'kdd', TmpDir = "./DataProcessor/data")
    raw_data, length = a.loadSparseDF(active_features=active, window_lengths=window_lengths)
    raw_data = raw_data.toarray()

    for k, v, in length.items():
        print(k, ": ", v)

    data =  raw_data[:, 4:]
    label =  raw_data[:, 3]

    data = to_dataset(data)
    label = to_dataset(label)
    
    dataset = tf.data.Dataset.zip((data, label))
    dataset = dataset.map(lambda data, label: (tf.cast(data, dtype=tf.float32), tf.cast(label, dtype=tf.float32)))
    
    train_dataset, test_dataset = split_dataset(dataset, train_fraction=0.8)
    train_dataset = train_dataset.padded_batch(batch_size, drop_remainder=False)
    test_dataset = test_dataset.padded_batch(batch_size, drop_remainder=False)

    return train_dataset, test_dataset

class KTM(tf.keras.Model):
    def __init__(self, feature_num,  embed_dim):
        super(KTM, self).__init__()
        self.embed = self.add_weight(name="embed", shape=(feature_num, embed_dim), initializer="uniform")
        self.bias = self.add_weight(name="bias", shape=(feature_num, 1), initializer="zeros")
        self.global_bias = self.add_weight(name="global_bias", shape=(1, 1), initializer="zeros")

        # train
        self.loss = tf.keras.losses.BinaryCrossentropy()
        self.opti = tf.keras.optimizers.Adam()
    
        # metrics
        self.metrics_loss = tf.keras.metrics.BinaryCrossentropy()
        self.metrics_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.metrics_auc = tf.keras.metrics.AUC()

        # link
        # probit
        # self.link_fun = tfp.distributions.Normal(loc=0., scale=1.).cdf
        # sigmoid
        self.link_fun = tf.keras.activations.sigmoid
        

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
    
    def loss_function(self, label, pred):
        loss = self.loss(label, pred)
        return loss

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
    
    def reset_metrics(self):
        self.metrics_loss.reset_states()
        self.metrics_accuracy.reset_states()
        self.metrics_auc.reset_states()
    
    def update_metrics(self, label, pred):
        self.metrics_loss.update_state(label, pred)
        self.metrics_accuracy.update_state(label, pred)
        self.metrics_auc.update_state(label, pred)

    @tf.function(experimental_relax_shapes=True)
    def metrics(self, dataset):
        self.reset_metrics()
        for data, label in dataset:
            pred = self(data)
            label = tf.expand_dims(label, axis=-1)
            pred = tf.expand_dims(pred, axis=-1)
            self.update_metrics(label, pred)
        loss = self.metrics_loss.result()
        acc = self.metrics_accuracy.result()
        auc = self.metrics_auc.result()
        self.reset_metrics()
        return loss, acc, auc

@tf.function
def train(epoch, model, train_dataset, test_dataset):
    element_num = tf.data.experimental.cardinality(train_dataset)
    for i, (data, label) in train_dataset.repeat(epoch).enumerate():
        model.train_step(data, label)
        if tf.equal(tf.math.floormod(i+1, element_num), 0):
            train_loss, train_acc, train_auc = model.metrics_loss.result(), model.metrics_accuracy.result(), model.metrics_auc.result()
            test_loss, test_acc, test_auc = model.metrics(test_dataset)
            tf.print("epoch: ", tf.math.floordiv(i, element_num), "train_loss: ", train_loss, "train_acc: ", train_acc, "train_auc: ", train_auc, 
                    "test_loss: ", test_loss, "test_acc: ", test_acc, "test_auc: ", test_auc)
            model.reset_metrics()
if __name__ == "__main__":
    tf.config.experimental_run_functions_eagerly(True)
    train_dataset, test_dataset = load_data(batch_size=128)
    feature_num = [d for d, l in train_dataset.take(1)][0].shape[-1]
    print(feature_num)
    
    model = KTM(feature_num=feature_num, embed_dim=20)
    train(epoch=30, model=model, train_dataset=train_dataset, test_dataset=test_dataset)
