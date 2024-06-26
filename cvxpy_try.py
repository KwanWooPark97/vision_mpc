import cvxpy as cvx
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from cartpole_gym_env import CartPoleEnv
import matplotlib.pyplot as plt

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
# 초기 조건들 설정 입력 먼저 초기 조건 설정
F_init = 0.0
env=CartPoleEnv("huma")
state,_=env.reset()
plt.figure(figsize=(8,5))
plt.ion()
plt.show()
plot_t=[]
force=0.0
state=np.append(state,force)
mma=[[-10],[10],[-100],[100],[-6.28],[6.28],[-100,100],[-10],[10],[-100],[100],[-6.28],[6.28],[-100,100]]
plot_x_hat=[]
plot_theta_hat=[]
state_deq = deque([np.zeros_like(state) for _ in range(10)],maxlen=10)
state_deq.append(state)
state_init=np.array(state_deq)
model=LSTM_test()
model.load_weights('sample_model3d3')
plot_force=[]
plot_x=[]
plot_theta=[]
t=0
while True:

    mpc_data=np.array(state_deq)
    x = np.array(state_deq).reshape([1, 10, 5])

    cart_position_hat, cart_position_dot_hat, theta_real_hat, theta_dot_real_hat = model.predict(x)[0]
    #m.time = [times, times + 0.1, times + 0.2, times + 0.3, times + 0.4, times + 0.5]
    cart_position, cart_position_dot, theta_real, theta_dot_real = env.step(force)
    #x_hat=cart_position_hat
    #x_dot_hat=cart_position_dot_hat
    #theta_hat=theta_real_hat
    #theta_dot_hat=theta_dot_real_hat

    #x_hat,x_dot_hat,theta_hat,theta_dot_hat=model.predict(data)
    next_state = np.array([cart_position_hat, cart_position_dot_hat, theta_real_hat, theta_dot_real_hat])
    u = cvx.Variable()
    obj = cvx.Minimize(cart_position_hat**2+cart_position_dot_hat**2+theta_real_hat**2+theta_dot_real_hat**2 + u**2)

    const=[-30<=u,u<=30]
    prob=cvx.Problem(obj,const)
    result=prob.solve()

    # retrieve new Tc value
    force =u.value
    plot_x.append(cart_position)
    plot_theta.append(theta_real)
    plot_force.append(force)
    plot_t.append(t)

    state = np.append(next_state, force)
    state_deq.append(state)
    t+=1
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(plot_t, plot_force, 'b--', lw=3)
    plt.ylabel('input Force')
    plt.legend(['force'], loc='best')

    plt.subplot(3, 1, 2)
    plt.plot(plot_t, plot_x, 'b.-', lw=3, label=r'$position$')
    # plt.plot(plot_t, np.zeros(plot_t), 'r-', lw=3, label=r'$position_{sp}$')
    plt.axhline(0.0, 0.1, 0.9, color='r', linestyle='-', label=r'$position_{sp}$')
    plt.ylabel(r'cart position')
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    # plt.plot(plot_t, np.zeros(plot_t), 'r-', lw=3, label=r'$theta_{sp}$')
    plt.axhline(0.0, 0.1, 0.9, color='r', linestyle='-', label=r'$position_{sp}$')
    plt.plot(plot_t, plot_theta, 'b.-', lw=3, label=r'$theta_{meas}$')
    plt.ylabel('theta')
    plt.xlabel('Time (min)')
    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.01)