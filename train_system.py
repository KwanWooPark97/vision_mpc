import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import math
import numpy as np
from scipy.integrate import odeint
from cpprb import ReplayBuffer #강화학습의 PER,HER,ReplayBuffer등을 구현해둔 라이브러리입니다.
from collections import deque #list 타입의 변수의 최대 길이를 정해주는 라이브러리입니다.

def get_default_rb_dict(size): #replaybuffer에 들어갈 요소들과 크기를 정해줍니다.
    return {
        "size": size,
        "default_dtype": np.float32,
        "env_dict": {
            "x": {                  #observation
                "shape": (10,5)},
            "next_x": {  # observation
                "shape": (4)}}}

def get_replay_buffer():

    kwargs = get_default_rb_dict(size=15000) #replaybuffer를 만들어줍니다. 최대 크기는 size로 정해줍니다.

    return ReplayBuffer(**kwargs)

class LSTM_test(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.s1 = LSTM(512, return_sequences=True)
        self.s2 = LSTM(256, return_sequences=True)
        self.s3 = LSTM(128)
        self.s4 = Dense(4)

    def call(self, states):
        features1 = self.s1(states)
        features1 = self.s2(features1)
        features1 = self.s3(features1)
        features1 = self.s4(features1)

        return features1



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
    replay_buffer = get_replay_buffer()
    replay_buffer.load_transitions('data_buffer.npz')  # 학습에 사용할 데이터를 가져옵니다.
    samples= replay_buffer.get_all_transitions(shuffle=True)  # replay_buffer에서 batch_size 만큼 sample을 가져옵니다.
    input_data,output_data= samples["x"], samples["next_x"]

    # Shuffle and batch data
    n_batch = 500

    callback = [tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                   monitor='val_loss',
                                                   save_best_only=True)]

    model = LSTM_test()
    opt_adam = Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss=tf.keras.losses.mse)  #

    # Train model
    n_epochs = 100
    history = model.fit(input_data,output_data, epochs=300, verbose=2)
    model.save_weights('sample_model3d2')