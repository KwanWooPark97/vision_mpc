import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

class LSTM_test(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.s1 = LSTM(512, return_sequences=True)
        self.s2 = LSTM(256, return_sequences=True)
        self.s3 = LSTM(52)

    def call(self, states):
        features1 = self.s1(states)
        features1 = self.s2(features1)
        features1 = self.s3(features1)

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
    # Load datasets
    train_data =
    valid_data =

    # Shuffle and batch data
    n_batch = 30

    callback = [tf.keras.callbacks.ModelCheckpoint(filepath='best_model.h5',
                                                   monitor='val_loss',
                                                   save_best_only=True)]

    model = LSTM_test()
    opt_adam = Adam(learning_rate=0.00001)
    model.compile(optimizer='adam', loss=tf.keras.losses.mse)  #

    # Train model
    n_epochs = 100
    history = model.fit(train_data, epochs=n_epochs, verbose=2, validation_data=valid_data)
    model.save_weights('sample_model3d')