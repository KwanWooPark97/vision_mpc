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
                "shape": (10,10)},
            "next_x": {  # observation
                "shape": (8)}}}

def get_replay_buffer():

    kwargs = get_default_rb_dict(size=100000) #replaybuffer를 만들어줍니다. 최대 크기는 size로 정해줍니다.

    return ReplayBuffer(**kwargs)


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gekko import GEKKO
import math
import datetime
from cartpole_gym_env import CartPoleEnv
import random
from collections import deque

#model=LSTM_test()
np.random.seed(1)
replay_buffer = get_replay_buffer()
n = 8
m = 2
T = 30
alpha = 0.2
beta = 3
A = np.eye(n) - alpha * np.random.rand(n, n)
B = np.random.randn(n, m)
x_0 = beta * np.random.randn(n)
u = np.zeros(m)
state=np.append(x_0,u)
state_deq = deque([np.zeros_like(state) for _ in range(10)],maxlen=10)
t=0

while replay_buffer.get_current_episode_len() <= 100000:
    u = np.array([random.uniform(-1, 1),random.uniform(-1,1)])
    next_state=A.dot(x_0)+B.dot(u)
    state = np.append(x_0, u)
    state_deq.append(state)

    replay_buffer.add(x=state_deq, next_x=next_state)
    # retrieve new Tc value
    t+=1
    if t%30==0:
        state_deq = deque([np.zeros_like(state) for _ in range(10)], maxlen=10)
        x_0 = beta * np.random.randn(n)
    else:
        x_0=next_state


replay_buffer.save_transitions("data_buffer_simple_big")