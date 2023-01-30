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
            "obs": {                  #observation
                "shape": (10,5)},
            "act": {
                "shape": (4)}}}

def get_replay_buffer():

    kwargs = get_default_rb_dict(size=15000) #replaybuffer를 만들어줍니다. 최대 크기는 size로 정해줍니다.

    return ReplayBuffer(**kwargs)
def pendulum(self, state, t, u):
    # Inputs (1):
    # Forc
    force = u

    # States (4):
    # x = state[0]
    # x_dot=state[1]
    theta = state[2]
    theta_dot = state[3]

    costheta = math.cos(math.radians(theta))
    sintheta = math.sin(math.radians(theta))
    gravity = 9.8
    masscart = 2.0
    masspole = 0.3
    total_mass = masspole + masscart
    length = 1.0  # actually half the pole's length
    polemass_length = masspole * length
    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    # Return xdot:
    xdot = np.zeros(4)
    xdot[3] = thetaacc
    xdot[2] = xdot[3]
    xdot[1] = xacc
    xdot[0] = xdot[1]

    return xdot

replay_buffer = get_replay_buffer()
force=0.0
state=[0.0,0.0,0.0,0.0]
time=0.0
while True:
    ts = [time,time+0.08]
    y = odeint(pendulum, state, ts, args=(force,))
    time+=0.08
    state[0] = y[-1][0]
    state[1] = y[-1][1]
    state[2] = y[-1][2]
    state[3] = y[-1][3]
