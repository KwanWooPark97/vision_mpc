import math
import tensorflow as tf
import numpy as np



class MPC_plant(object):

    def __init__(self):
        self.h_t=30
        self.w2=0.1
        self.cb1=24.9
        self.cb2=0.1
        self.k1=1.0
        self.k2=1.0











if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)