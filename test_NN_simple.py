import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gekko import GEKKO
import math
import datetime
from cartpole_gym_env import CartPoleEnv
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
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

class LSTM_test(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.s1 = LSTM(128, return_sequences=True)
        self.s2 = LSTM(64, return_sequences=True)
        self.s3 = LSTM(32)
        self.s4 = Dense(8)

    def call(self, states):
        features1 = self.s1(states)
        features1 = self.s2(features1)
        features1 = self.s3(features1)
        features1 = self.s4(features1)

        return features1
np.random.seed(1)

# Simulate CSTR
plot_force=[]
plot_x=[]
plot_theta=[]
plt.figure(figsize=(8,5))
plt.ion()
plt.show()
plot_t=[]
t=0
times=0
model=LSTM_test()
model.load_weights('sample_model3d_simple')
n = 8
m = 2
T = 50
alpha = 0.2
beta = 3
A = np.eye(n) - alpha * np.random.rand(n, n)
B = np.random.randn(n, m)
x_0 = beta * np.random.randn(n)
u = np.zeros(m)
state=np.append(x_0,u)
state_deq = deque([np.zeros_like(state) for _ in range(10)],maxlen=10)

plot_x_hat=[]
plot_theta_hat=[]
while t<=30:
    u = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
    next_state = A.dot(x_0) + B.dot(u)
    state = np.append(x_0, u)

    state_deq.append(state)
    input_data=np.array(state_deq).reshape(1,10,10)
    next_state_hat = model(input_data)[0]
    # retrieve new Tc value
    t += 1

    plot_x.append(next_state[0])
    plot_x_hat.append(next_state_hat[0])
    plot_theta.append(next_state[1])
    plot_theta_hat.append(next_state_hat[1])
    plot_force.append(u[0])
    plot_t.append(t)
    x_0=next_state

    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(plot_t, plot_force, 'b--', lw=3)
    plt.ylabel('input Force')
    plt.legend(['force'], loc='best')

    plt.subplot(3, 1, 2)
    plt.plot(plot_t, plot_x, 'b.-', lw=3, label=r'$position$')
    plt.plot(plot_t, plot_x_hat, 'g-', lw=3, label=r'$NN$')
    plt.axhline(0.0, 0.1, 0.9, color='r', linestyle='-', label=r'$position_{sp}$')
    plt.ylabel(r'cart position')
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    plt.plot(plot_t, plot_theta_hat, 'g-', lw=3, label=r'$NN$')
    plt.axhline(0.0, 0.1, 0.9, color='r', linestyle='-', label=r'$theta_{sp}$')
    plt.plot(plot_t, plot_theta, 'b.-', lw=3, label=r'$theta_{meas}$')
    plt.ylabel('theta')
    plt.xlabel('Time (min)')
    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.01)

