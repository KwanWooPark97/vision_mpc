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
                "shape": (10, 4)}}}

def get_replay_buffer():

    kwargs = get_default_rb_dict(size=15000) #replaybuffer를 만들어줍니다. 최대 크기는 size로 정해줍니다.

    return ReplayBuffer(**kwargs)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gekko import GEKKO
import math
import datetime
from cartpole_gym_env import CartPoleEnv
import random

# Simulate CSTR
plot_force=[]
plot_x=[]
plot_theta=[]
t=[]
force=0.0
env=CartPoleEnv("huma")
_,_=env.reset()
plt.figure(figsize=(8,5))
plt.ion()
plt.show()
plot_t=[]
t=0
times=0
#model=LSTM_test()
replay_buffer = get_replay_buffer()
while True:
    #m.time = [times, times + 0.1, times + 0.2, times + 0.3, times + 0.4, times + 0.5]
    cart_position, cart_position_dot, theta_real, theta_dot_real = env.step(force)
    replay_buffer.add(x=state_deq, act=action_deq, rtg=reward_deq, traj_mask=traj_mask_deq, time_step=time_step_deq)
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
