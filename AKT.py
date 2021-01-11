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

def fake_data():
    pid_num = 10
    cid_num = 5
    Q = [[0, 0, 1, 1, 0],[1, 0, 1, 0, 0],[0, 0, 1, 0, 1],[0, 0, 0, 0, 1],[0, 0, 1, 0, 0]]
    time = [[1, 2, 4, 7, 10, 13],
        [1, 3, 4, 6, 0, 0]]
    pid = [[8, 2, 4, 3, 10, 5],
        [7, 3, 4, 6, 0, 0]]

    res = [[0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 0]]
    
    time = tf.constant(time, dtype=tf.float32)
    pid = tf.constant(pid, dtype=tf.float32)
    res = tf.constant(res, dtype=tf.float32)
    Q = tf.constant(Q, dtype=tf.int32)
    return pid, res, time, Q, pid_num, cid_num

class Encoder(tf.keras.layers.Layer):
    def __init__(self, model_name, num_head, dim):
        super(Encoder, self).__init__()
        self.model_name = model_name
        self.num_head = num_head
        self.dim = dim
        self.wQ = self.add_weight(name=self.model_name + "wQ", shape=(self.num_head, self.dim, self.dim))
        self.wK = self.add_weight(name=self.model_name + "wK", shape=(self.num_head, self.dim, self.dim))
        self.wV = self.add_weight(name=self.model_name + "wV", shape=(self.num_head, self.dim, self.dim))
        self.wO = self.add_weight(name=self.model_name + "wO", shape=(self.num_head * self.dim, self.dim))
        self.theta = self.add_weight(name=self.model_name + "theta", shape=(1, 1))

    def time_factor(self, x, timestamp, low_tri):
        x = tf.math.cumsum(x, axis=-1)
        x = 1 - x
        diag_timestamp = tf.expand_dims(tf.linalg.diag_part(timestamp), axis=-1)
        d_t = diag_timestamp - timestamp
        d = x * tf.cast(d_t, dtype=tf.float32) * low_tri
        return d

    def get_tri_mask(self, seq_len):
        low_tri = tf.linalg.band_part(tf.ones((seq_len, seq_len)),-1, 0)
        return low_tri

    def attention(self, Q, K, V, timestamp):
        timestamp = tf.expand_dims(timestamp, axis=-2)
        timestamp = tf.tile(timestamp, multiples=[1, tf.shape(timestamp)[-1], 1])
        exp_x = tf.divide(tf.matmul(Q, K, transpose_b=True), tf.sqrt(tf.cast(self.dim, dtype=tf.float32)))
        x = tf.keras.activations.softmax(exp_x)
        seq_len = tf.shape(x)[-1]
        low_tri = self.get_tri_mask(seq_len)
        x = x * low_tri
        # normalize
        x = tf.math.divide_no_nan(x, tf.reduce_sum(x, axis=-1, keepdims=True))
        d = self.time_factor(x, timestamp, low_tri)
        s = exp_x * tf.exp(-1 * self.theta * self.theta * d)
        a = tf.keras.activations.softmax(s)
        a = a * low_tri
        # normalize
        a = tf.math.divide_no_nan(a, tf.reduce_sum(a, axis=-1, keepdims=True))
        ret = a @ V
        return ret
    
    def multi_attention(self, inputs, timestamp):
        output = tf.TensorArray(dtype=tf.float32, size=self.num_head)
        for i in tf.range(self.num_head):
            Q = inputs @ self.wQ[i]
            K = inputs @ self.wK[i]
            V = inputs @ self.wV[i]
            ret = self.attention(Q, K, V, timestamp)
            output = output.write(i, ret)
        output = output.stack()
        output = tf.transpose(output, perm=[1, 2, 0, 3])
        output_shape = tf.concat([tf.shape(output)[:2], [-1]], axis=-1)
        output = tf.reshape(output, shape=(output_shape))
        output = tf.matmul(output, self.wO)
        return output


class KnowledgeRetriever(Encoder):
    def __init__(self, model_name, num_head, dim):
        super(KnowledgeRetriever, self).__init__(model_name, num_head, dim)
    
    def get_tri_mask(self, seq_len):
        low_tri = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), 0, -1)
        return low_tri

    def multi_attention(self, x, y, timestamp):
        output = tf.TensorArray(dtype=tf.float32, size=self.num_head)
        for i in tf.range(self.num_head):
            Q = x @ self.wQ[i]
            K = x @ self.wK[i]
            V = y @ self.wV[i]
            ret = self.attention(Q, K, V, timestamp)
            output = output.write(i, ret)
        output = output.stack()
        output = tf.transpose(output, perm=[1, 2, 0, 3])
        output_shape = tf.concat([tf.shape(output)[:2], [-1]], axis=-1)
        output = tf.reshape(output, shape=(output_shape))
        output = tf.matmul(output, self.wO)
        return output

class AKT(tf.keras.Model):
    def __init__(self, model_params):
        super(AKT, self).__init__()
        self.problem_num = model_params['problem_num']
        self.concept_num = model_params['concept_num']
        self.Q = model_params['Q_matrix']

        self.embed_dim = model_params['embed_dim']
        self.num_head = model_params['num_head']

        self.c_embed = self.add_weight(name="concept_embed", shape=(self.concept_num, self.embed_dim))
        self.d_embed = self.add_weight(name="difficult_embed", shape=(self.concept_num, self.embed_dim))
        self.f_embed = self.add_weight(name="f_embed", shape=(self.concept_num, self.embed_dim))
        self.mu_q = self.add_weight(name="q_difficult", shape=(self.concept_num, 1))
        self.r_embed = self.add_weight(name="response_embed", shape=(2, self.embed_dim))

        self.d = tf.keras.layers.Dense(units=1, activation='sigmoid')

        self.q_encoder = Encoder("q_encoder", self.num_head, self.embed_dim)
        self.k_encoder = Encoder("k_encoder", self.num_head, self.embed_dim)

        self.knowledge_retriever = KnowledgeRetriever("knowledge_retriver", self.num_head, self.embed_dim)

        self.loss_obj = tf.keras.losses.BinaryCrossentropy()
        # opti
        self.opti = tf.keras.optimizers.Adam()
        # metrics
        self.metrics_format = "epoch,time,train_loss,train_acc,train_pre,train_rec,train_auc,train_mae,train_rmse,test_loss,test_acc,test_pre,test_rec,test_auc,test_mae,test_rmse"
        self.metrics_path = 'file://' + model_params['metrics_path']

        # metrics
        self.metrics_loss = tf.keras.metrics.BinaryCrossentropy()
        self.metrics_acc = tf.keras.metrics.BinaryAccuracy()
        self.metrics_pre = tf.keras.metrics.Precision()
        self.metrics_rec = tf.keras.metrics.Recall()
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
    def call(self, inputs):
        item, timestamp, correct = tf.split(inputs, num_or_size_splits=3, axis=-1)

        item = tf.cast(item, dtype=tf.int32)
        correct = tf.cast(correct, dtype=tf.int32)
        item = tf.squeeze(item, axis=-1)
        item = item - 1
        timestamp = tf.squeeze(timestamp, axis=-1)
        correct = tf.squeeze(correct, axis=-1)
        item_one_hot = tf.one_hot(tf.cast(item, dtype=tf.int32), depth=self.problem_num)
        concept = tf.matmul(item_one_hot, self.Q)
        mu_q = tf.reshape(self.mu_q, shape=(1, self.concept_num))
        concept_mu_q = concept * mu_q
        seq_concept_embed = tf.matmul(concept, self.c_embed)
        # mu_q_t @ d_c_t + C_c_t
        x = concept_mu_q @ self.d_embed + seq_concept_embed
        concept_num = tf.reduce_sum(concept, axis=-1, keepdims=True)
        correct_one_hot = tf.one_hot(correct, depth=2)
        seq_r_embed = tf.matmul(correct_one_hot, self.r_embed)

        y = seq_r_embed * concept_num + concept_mu_q @ self.f_embed

        x_hat = self.q_encoder.multi_attention(x, timestamp)
        y_hat = self.k_encoder.multi_attention(y, timestamp)
        output = self.knowledge_retriever.multi_attention(x_hat, y_hat, timestamp)
        output = output[:, 1:, :]
        output_shape = tf.shape(output)
        pad_shape = tf.concat([output_shape[:1], [1], output_shape[2:]], axis=-1)
        pad = tf.zeros(pad_shape)
        output = tf.concat([pad, output], axis=1)
        # predict layer
        pred = tf.concat([output, x_hat], axis=-1)
        pred = self.d(pred)
        return pred

    def get_mask(self, data):
        item, timestamp, correct = tf.split(data, num_or_size_splits=3, axis=-1)
        item = tf.squeeze(item, axis=-1)
        mask = tf.cast(tf.not_equal(item, 0), dtype=tf.float32)
        return mask

    def loss_function(self, pred, label, mask):
        label = tf.expand_dims(label, axis=-1)
        loss = self.loss_obj(label, pred, mask)
        self.update_metrics(label, pred, mask)
        return loss

    def update_metrics(self, label, pred, mask):
        # metrics
        self.metrics_loss.update_state(label, pred, mask)
        self.metrics_acc.update_state(label, pred, mask)
        self.metrics_pre.update_state(label, pred, mask)
        self.metrics_rec.update_state(label, pred, mask)
        self.metrics_auc.update_state(label, pred, mask)
        self.metrics_mae.update_state(label, pred, mask)
        self.metrics_rmse.update_state(label, pred, mask)

    @tf.function(experimental_relax_shapes=True)
    def metrics_step(self, data, label):
        pred = self(data)
        mask = self.get_mask(data)
        self.loss_function(pred, label, mask)

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, data, label):
        with tf.GradientTape() as tape:
            pred = self(data)
            mask = self.get_mask(data)
            loss = self.loss_function(pred, label, mask)
        grad = tape.gradient(loss, self.trainable_variables)
        self.opti.apply_gradients(zip(grad, self.trainable_variables))

def get_last_epoch_data(model, dataset):
    all_pred = list()
    all_label = list()
    model.resetMetrics()
    for data, label in dataset:
        pred = model.metrics_step(data, label)
    return (model.metrics_acc.result(), model.metrics_pre.result(), model.metrics_rec.result(), 
    model.metrics_auc.result(), model.metrics_mae.result(), model.metrics_rmse.result())
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

            tf.print("epoch: ", tf.math.floordiv(i+1, element_num), "time: ", end - start,
                    "train_loss: ", train_loss,  "train_acc: ", train_acc,
                    "train_pre: ", train_pre, "train_rec: ", train_rec,
                    "train_auc: ", train_auc, "train_mae: ", train_mae,
                    "train_rmse: ", train_rmse,
                    "test_loss: ", test_loss,  "test_acc: ", test_acc,
                    "test_pre: ", test_pre, "test_rec: ", test_rec,
                    "test_auc: ", test_auc, "test_mae: ", test_mae,
                    "test_rmse: ", test_rmse)
            start = tf.timestamp()

def runOJ(fold_id, is_test=True):
    #######################################
    # LC parameters
    #######################################
    userLC = [30, 3600, 0.1, 1]
    problemLC = [30, 1e9, 0, 1]
    # algebra08原始数据里的最值，可以注释，不要删
    low_time = "2018-06-01 00:00:00"
    high_time = "2018-11-29 00:00:00"
    timeLC = [low_time, high_time]
    data_processor = _DataProcessor(userLC, problemLC, timeLC, 'oj', TmpDir = "./DataProcessor/data")

    LCDataDir = data_processor.LCDataDir
    saveDir = os.path.join(LCDataDir, 'AKT')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    LC_params = data_processor.LC_params

    dataset_params = copy.deepcopy(LC_params)
    dataset_params["trainRate"] = 0.8
    dataset_params["batch_size"] = 4
    dataset_params["kFold"] = 5
    [train_dataset, test_dataset, Q_matrix] = data_processor.loadAKTData_5F(dataset_params, fold_id)
    Q_matrix = Q_matrix.toarray().astype(np.float32)

    #######################################
    # model parameters
    #######################################
    model_params = {}
    model_params['problem_num'] = Q_matrix.shape[0]
    model_params['concept_num'] = Q_matrix.shape[1]
    model_params['embed_dim'] = 20
    model_params['epoch'] = 200
    model_params['threshold'] = 0.5
    model_params['metrics_path'] = saveDir + '/metrics.csv'
    model_params["data_shape"] = [data for data, label in train_dataset.take(1)][0].shape.as_list()
    model_params['Q_matrix'] = Q_matrix
    model_params['num_head'] = 5


    model = AKT(model_params)
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
    model_params.pop("Q_matrix")
    # saveDict(results, saveDir, 'results'+ getLegend(model_params)+'.json')

def runKDD(fold_id, is_test=True):
    #######################################
    # LC parameters
    #######################################
    userLC = [30, 3600]
    problemLC = [30, 1e9]
    # algebra08原始数据里的最值，可以注释，不要删
    low_time = "2008-09-08 14:46:48"
    high_time = "2009-01-01 00:00:00"
    timeLC = [low_time, high_time]
    data_processor = _DataProcessor(userLC, problemLC, timeLC, 'kdd', TmpDir = "./DataProcessor/data")

    LCDataDir = data_processor.LCDataDir
    saveDir = os.path.join(LCDataDir, 'AKT')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    LC_params = data_processor.LC_params

    dataset_params = copy.deepcopy(LC_params)
    dataset_params["trainRate"] = 0.8
    dataset_params["batch_size"] = 32
    dataset_params["kFold"] = 5
    [train_dataset, test_dataset, Q_matrix] = data_processor.loadAKTData_5F(dataset_params, fold_id)
    Q_matrix = Q_matrix.toarray().astype(np.float32)

    #######################################
    # model parameters
    #######################################
    model_params = {}
    model_params['problem_num'] = Q_matrix.shape[0]
    model_params['concept_num'] = Q_matrix.shape[1]
    model_params['embed_dim'] = 20
    model_params['epoch'] = 200
    model_params['threshold'] = 0.5
    model_params['metrics_path'] = saveDir + '/metrics.csv'
    model_params["data_shape"] = [data for data, label in train_dataset.take(1)][0].shape.as_list()
    model_params['Q_matrix'] = Q_matrix
    model_params['num_head'] = 5

    model = AKT(model_params)
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
    model_params.pop("Q_matrix")
    #saveDict(results, saveDir, 'results' + getLegend(model_params) + '.json')

def runAssist(fold_id, is_test=True):
    #######################################
    # LC parameters
    #######################################
    userLC = [10, 3000]
    problemLC = [10, 3000]
    #hdu原始数据里的最值，可以注释，不要删
    low_time = "2012-09-01 00:00:00" 
    high_time = "2012-09-30 00:00:00"
    timeLC = [low_time, high_time]
    data_processor = _DataProcessor(userLC, problemLC, timeLC, 'assist', TmpDir = "./DataProcessor/data")


    LCDataDir = data_processor.LCDataDir
    saveDir = os.path.join(LCDataDir, 'AKT')
    print("===================================")
    print("metrics save path: ", saveDir)
    print("===================================")
    prepareFolder(saveDir)
    LC_params = data_processor.LC_params

    dataset_params = copy.deepcopy(LC_params)
    dataset_params["trainRate"] = 0.8

    dataset_params["batch_size"] = 32
    dataset_params['kFold'] = 5
    [train_dataset, test_dataset, Q_matrix] = data_processor.loadAKTData_5F(dataset_params, fold_id)
    Q_matrix = Q_matrix.toarray().astype(np.float32)

    #######################################
    # model parameters
    #######################################
    model_params = {}
    model_params['problem_num'] = Q_matrix.shape[0]
    model_params['concept_num'] = Q_matrix.shape[1]
    model_params['embed_dim'] = 20
    model_params['epoch'] = 200
    model_params['threshold'] = 0.5
    model_params['metrics_path'] = saveDir + '/metrics.csv'
    model_params["data_shape"] = [data for data, label in train_dataset.take(1)][0].shape.as_list()
    model_params['Q_matrix'] = Q_matrix
    model_params['num_head'] = 5


    model = AKT(model_params)
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
    #saveDict(results, saveDir, 'results'+ getLegend(model_params)+'.json')
def set_run_eagerly(is_eager=False):
    if tf.__version__ == "2.2.0":
        tf.config.experimental_run_functions_eagerly(is_eager)
    else:
        tf.config.run_functions_eagerly(is_eager)
if __name__ == "__main__":
    tf.debugging.enable_check_numerics()
    set_run_eagerly(False)
    #runAssist(0, False)
    runKDD(0, False)
    runOJ(0, False)
