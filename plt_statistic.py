import os
import sys
import tensorflow as tf
sys.path.append("./DataProcessor/")
from DataProcessor import _DataProcessor
import matplotlib.pyplot as plt

def runAssist(is_test=True):
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
    split_dict_data = data_processor.loadSplitInfo(kFold=5)
    [df, QMatrix, StaticInformation, DictList] = data_processor.dataprocessor.loadLCData()


    # others
    all_user = list(df["user_id"].unique())
    num_user = len(all_user)
    train_uid = all_user[:int(num_user * 0.8)]
    test_uid = all_user[int(num_user * 0.8):]
    plt.title("no shuffle")
    train_length = [len(df[df['user_id'] == uid]) for uid in train_uid]
    test_length = [len(df[df['user_id'] == uid]) for uid in test_uid]
    plt.hist(train_length, bins=20, density=True, label="train")
    plt.hist(test_length, bins=20, density=True, label="test")
    plt.legend()
    plt.show()

    for i in range(5):
        train_uid = split_dict_data[str(i)]['train']
        train_length = [len(df[df['user_id'] == uid]) for uid in train_uid]
        plt.title("FOLD: " + str(i))
        plt.hist(train_length, bins=20, density=True, label="train")
        test_uid = split_dict_data[str(i)]['test']
        test_length = [len(df[df['user_id'] == uid]) for uid in test_uid]
        plt.hist(test_length, bins=20, density=True, label="test")
        plt.legend()
        plt.show()
        


    
def set_run_eagerly(is_eager=False):
    if tf.__version__ == "2.2.0":
        tf.config.experimental_run_functions_eagerly(is_eager)
    else:
        tf.config.run_functions_eagerly(is_eager)
if __name__ == "__main__":
    set_run_eagerly(True)
    runAssist(False)
