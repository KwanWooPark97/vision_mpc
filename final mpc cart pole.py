import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gekko import GEKKO
import math
import datetime
from cartpole_gym_env import CartPoleEnv
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


class LSTM_test(tf.keras.Model):#[batch,step,features]
    def __init__(self):
        super().__init__()
        self.s1 = LSTM(512, return_sequences=True)
        self.s2 = LSTM(256, return_sequences=True)
        self.s3 = LSTM(4)

    def call(self, states):
        features1 = self.s1(states)
        features1 = self.s2(features1)
        features1 = self.s3(features1)

        return features1
# 초기 조건들 설정 입력 먼저 초기 조건 설정
F_init = 0.0

# Steady State Initial Conditions for the States
cart_position_init=0
cart_position_dot_init = 0
theta_init=0
theta_dot_init=0
x0 = np.empty(4)
x0[0] = cart_position_init
x0[1] = cart_position_dot_init
x0[2] = theta_init
x0[3] = theta_dot_init

#%% GEKKO nonlinear MPC
m = GEKKO(remote=False)
#m.time = [0,0.08,0.16,0.24,0.32,0.40,0.48,0.56,0.64]
#m.time = [0,0.05,0.1,0.15]

pi=math.pi

F = m.Var(value=0.0)
x = m.Var(value=cart_position_init,lb=-4.8,ub=4.8)
x_dot=m.Var(value=cart_position_dot_init)
theta=m.Var(value=0.0)
theta_dot=m.Var(value=0)
x_hat = m.Var(value=cart_position_init,lb=-4.8,ub=4.8)
x_dot_hat=m.Var(value=cart_position_dot_init)
theta_hat=m.Var(value=0.0)
theta_dot_hat=m.Var(value=0)
m.Obj(((x-x_hat)+(x_dot-x_dot_hat)+(theta-theta_hat)+(theta_dot-theta_dot_hat))**2)

m.options.IMODE = 6
#m.options.SOLVER = 3
force=0.0
env=CartPoleEnv("human")
_,_=env.reset()
t=0
times=0
while True:
    #m.time = [times, times + 0.08, times + 0.16, times + 0.24, times + 0.32, times + 0.4]
    cart_position, cart_position_dot, theta_real, theta_dot_real = env.step(force)
    x = cart_position
    x_dot = cart_position_dot
    theta = theta_real
    theta_dot = theta_dot_real
    m.theta.MEAS = theta
    m.solve(disp=False)

    # retrieve new Tc value
    force =F.value[0]
    times+=0.08