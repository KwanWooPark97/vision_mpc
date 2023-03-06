from scipy.optimize import minimize
import numpy as np
from scipy.integrate import odeint
from gekko import GEKKO
import math
import datetime
from cartpole_gym_env import CartPoleEnv
import random
from collections import deque
import matplotlib.pyplot as plt
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
n = 8
m = 2
T = 30
alpha = 0.2
beta = 3
A = np.eye(n) - alpha * np.random.rand(n, n)
B = np.random.randn(n, m)
x_0 = beta * np.random.randn(n)
model=LSTM_test()
model.load_weights('sample_model3d_simple')
x = np.zeros(n)
u = np.zeros(m)
x_plot = np.zeros((n, T))
u_plot = np.zeros((m, T))


def cost_function(u, x):
    state_buffer=deque(np.array(x),maxlen=10)
    x_pred = np.zeros((n, T + 1))
    x_pred[:, 0] = x[-1][:8]
    u = u.reshape((m, T))
    for i in range(T):
        if i==0:
            input_data=np.array(state_buffer).reshape(1,10,10)
            x_pred[:, i + 1] = model(input_data)[0]
        else:
            data=np.append(x_pred[:, i],u[:,i])
            state_buffer.append(data)
            input_data=np.array(state_buffer).reshape(1,10,10)
            x_pred[:, i + 1] = model(input_data)[0]
    cost = 0
    for i in range(T):
        cost += np.sum(np.square(x_pred[:, i + 1])) + np.sum(np.square(u[:, i]))
    return cost


def constraint_function(u):
    return 1 - np.linalg.norm(u, ord=np.inf)

state=np.append(x_0,u)
plot_force=[]
plot_x=[]
plot_theta=[]
t=0
plot_x_hat=[]
plot_theta_hat=[]
state_deq = deque([np.zeros_like(state) for _ in range(10)],maxlen=10)
state_deq.append(state)
force_deq=deque([np.zeros_like(u) for _ in range(T)],maxlen=T)
#force_deq_mpc=deque([np.zeros_like(force) for _ in range(prediction_horizon)],maxlen=prediction_horizon)
con = {'type': 'ineq', 'fun': constraint_function}
# flatten x and u into a 1D array for optimization
u_control=np.zeros((m, T))
for i in range(T):
    print(i)
    res = minimize(cost_function, u_control, args=(state_deq,), constraints=con)
    #u_plot = u_plot.reshape((m, T))
    result = res.x.reshape((m, T))
    state=np.append(x_0,result[:,0])
    state_deq.append(state)
    u_plot[:, i] = result[:, 0]
    input_data = np.array(state_deq).reshape(1, 10, 10)
    x_0= model(input_data)[0]
    #x_0 = A.dot(x_0) + B.dot(result[:, 0])
    x_plot[:, i] = x_0
    u_control=np.concatenate((result[:,1:],result[:,-1].reshape(2,1)),axis=1)
#%config InlineBackend.figure_format = 'svg'

f = plt.figure()
u_plot=u_plot.reshape((m,T))
# Plot (u_t)_1.
ax = f.add_subplot(411)
plt.plot(u_plot[0, :])
plt.ylabel(r"$(u_t)_1$", fontsize=16)
plt.yticks(np.linspace(-1.0, 1.0, 3))
plt.xticks([])

# Plot (u_t)_2.
plt.subplot(4, 1, 2)
plt.plot(u_plot[1, :])
plt.ylabel(r"$(u_t)_2$", fontsize=16)
#plt.yticks(np.linspace(-1, 1, 3))
plt.xticks([])

# Plot (x_t)_1.
plt.subplot(4, 1, 3)
#x1 = x[0, :].value
plt.plot(x_plot[0,:])
plt.ylabel(r"$(x_t)_1$", fontsize=16)
plt.yticks([-10, 0, 10])
#plt.ylim([-10, 10])
plt.xticks([])

# Plot (x_t)_2.
plt.subplot(4, 1, 4)
#x2 = x[1, :].value
plt.plot(x_plot[1,:])
plt.yticks([-25, 0, 25])
#plt.ylim([-25, 25])
plt.ylabel(r"$(x_t)_2$", fontsize=16)
plt.xlabel(r"$t$", fontsize=16)
plt.tight_layout()
plt.show()