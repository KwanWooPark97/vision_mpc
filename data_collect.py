import math
import numpy as np
from scipy.integrate import odeint

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

force=0.0
state=[0.0,0.0,0.0,0.0]
time=0.0
while True:
    ts = [time,time+0.08]
    y = odeint(pendulum, state, ts, args=(force,))
    time+=0.08