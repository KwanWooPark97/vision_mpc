import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gekko import GEKKO
import math
import datetime
from cartpole_gym_env import CartPoleEnv
import random
#import tensorflow as tf
#from tensorflow.keras.layers import LSTM, Dense
#from tensorflow.keras.optimizers import Adam


'''gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)'''


'''class LSTM_test(tf.keras.Model):#[batch,step,features]
    def __init__(self):
        super().__init__()
        self.s1 = LSTM(512, return_sequences=True)
        self.s2 = LSTM(256, return_sequences=True)
        self.s3 = LSTM(4)

    def call(self, states):
        features1 = self.s1(states)
        features1 = self.s2(features1)
        features1 = self.s3(features1)

        return features1'''
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
Kd=10
#z_dot=x_dot
mPend=1.0
L=0.5
g=9.8
mCart=1.0
pi=math.pi

m.F = m.MV(value=0.0)
m.x = m.CV(value=cart_position_init,lb=-4.8,ub=4.8)
m.x_dot=m.CV(value=0.0)
m.theta=m.CV(value=-3.14)
m.theta_dot=m.CV(value=0.0)
'''x_hat = m.Var(value=0.0,lb=-4.8,ub=4.8)
x_dot_hat=m.Var(value=0.0)
theta_hat=m.Var(value=0.0)
theta_dot_hat=m.Var(value=0)'''
#thetaacc=m.Var(value=0.0000001)

#m.Equation(thetaacc==)
#m.Equation(m.temp == (m.F + polemass_length * m.theta_dot ** 2 * m.sin(m.theta*math.pi/180)) / total_mass)
m.Equation(m.x.dt()==m.x_dot)
m.Equation(m.x_dot.dt() ==(m.F - Kd*m.x_dot - mPend*L*m.theta_dot**2*m.sin(m.theta) + mPend*g*m.sin(m.theta)*m.cos(m.theta)) / (mCart + mPend*m.sin(m.theta)**2))
m.Equation(m.theta.dt()==m.theta_dot)
m.Equation(m.theta_dot.dt()==((m.F - Kd*m.x_dot - mPend*L*m.theta_dot**2*m.sin(m.theta))*m.cos(m.theta)/(mCart + mPend) + g*m.sin(m.theta)) / (L - mPend*L*m.cos(m.theta)**2/(mCart + mPend)))
#m.Obj(((x-x_hat)+(x_dot-x_dot_hat)+(theta-theta_hat)+(theta_dot-theta_dot_hat))**2+F**2)
#m.Obj(m.F**2)
#MV tuning
m.F.STATUS = 1
m.F.FSTATUS = 1
#m.F.DMAX = 10
#m.F.DMAXHI = 10   # constrain movement up
#m.F.DMAXLO = -10 # quick action down

#CV tuning
m.x.STATUS = 1
m.x.FSTATUS = 1
m.x.TR_INIT = 0
m.x.TAU = 0.1
m.x.SP=0

m.x_dot.STATUS = 1
m.x_dot.FSTATUS = 0
m.x_dot.TR_INIT = 0
m.x_dot.TAU = 0.1
m.x_dot.SP=0

m.theta.STATUS = 1
m.theta.FSTATUS = 1
m.theta.TR_INIT = 0
m.theta.TAU = 0.1
m.theta.SP=0

m.theta_dot.STATUS = 1
m.theta_dot.FSTATUS = 0
m.theta_dot.TR_INIT = 0
m.theta_dot.TAU = 0.1
m.theta_dot.SP=0

#m.Obj((m.x.SP-m.x)**2+(m.theta.SP-m.theta)**2+m.F**2)
m.options.CV_TYPE = 2
m.options.IMODE = 6
m.options.SOLVER = 3
force=0.0
env=CartPoleEnv("human")
_,_=env.reset()
# Simulate CSTR
plot_force=[]
plot_x=[]
plot_theta=[]
t=[]
env=CartPoleEnv("huma")
_,_=env.reset()
plt.figure(figsize=(8,5))
plt.ion()
plt.show()
plot_t=[]
t=0
times=0
#model=LSTM_test()
while True:
    #m.time = [times, times + 0.1, times + 0.2, times + 0.3, times + 0.4, times + 0.5]
    cart_position, cart_position_dot, theta_real, theta_dot_real = env.step(force)
    #m.x.MEAS=cart_position
    #m.x_dot.MEAS=cart_position_dot
    #m.theta.MEAS=theta_real
    #m.theta_dot.MEAS=theta_dot_real

    #x_hat,x_dot_hat,theta_hat,theta_dot_hat=model.predict(data)

    #m.solve(disp=False)

    # retrieve new Tc value
    force =random.uniform(-30,30)
    times+=0.1
    plot_x.append(cart_position)
    plot_theta.append(theta_real)
    plot_force.append(force)
    plot_t.append(t)
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