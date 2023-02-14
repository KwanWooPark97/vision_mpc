import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from collections import deque
from cartpole_gym_env import CartPoleEnv
import matplotlib.pyplot as plt
from tensorflow.keras import optimizers
from scipy.optimize import minimize
class LSTM_test(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.s1 = LSTM(512,return_sequences=True)
        self.s2 = LSTM(256, return_sequences=True)
        self.s3 = LSTM(128)
        self.s4 = Dense(4)

    def call(self, states):
        features1 = self.s1(states)
        features1 = self.s2(features1)
        features1 = self.s3(features1)
        features1 = self.s4(features1)

        return features1

model=LSTM_test()
model.load_weights('sample_model3d3')

# Define a cost function
'''def cost_function(u, model, x0, horizon):
    x_pred = x0
    cost = 0
    state_deqq=deque([np.array(x_pred[0][i]) for i in range(10)],maxlen=10)
    for t in range(horizon):
        next_states = model.predict(x_pred)[0]
        cost += next_states[0]**2 + next_states[2]**2 + u[t]**2
        states = np.append(next_states, u[t])
        state_deqq.append(states)
        x_pred = np.array(state_deqq).reshape([1, 10, 5])
    return cost'''

def cost_function(x, u,u0, x_pred):
    cost = 0
    T=10
    x_state=deque(x,maxlen=10)
    u_buffer=u0
    for t in range(T-1):
        u_buffer.append(u[t])
        input_data=np.concatenate((x[t], u_buffer), axis=1).reshape([1, 10, 5])
        # Use the deep learning model to predict the next state.
        #input_data=np.array(x_state[t]).reshape([1, 10, 5])
        x_pred[t+1,:] = model(input_data)
        # Accumulate the cost based on the deviation from the desired state and control inputs.
        cost += 0.1 * np.linalg.norm(x_pred[t+1,:] - np.array([0, 0, 0, 0]))**2 + 0.01 * np.linalg.norm(u[t,:])**2
        return cost

# Define the MPC function
def mpc_controller(x0,u0):


    jk=0
    #Use a solver to minimize the cost function subject to constraints.
    res = minimize(fun=cost_function, x0=u, args=(u,), method='L-BFGS-B')
    x_opt = res.x
    for i in range(T-1):
    def closure():
        nonlocal x, u, x_pred
        cost = cost_function(x, u,u0, x_pred)
        solver.minimize(cost, var_list=[u])
        return cost
    solver.minimize(closure, var_list=[u])
    # Apply the control inputs and update the state.
    u0.append(u[i])
    x0.append(x[i])
    input_data=np.concatenate((x0, u0), axis=1).reshape([1, 10, 5])
    x[i+1,:] = model(input_data)[0]
    x_pred[i+1,:] = x[i+1,:]
    # Return the first control input.
    return u[0,:]

horizon = 5
control_bounds = [-30, 30]
env=CartPoleEnv("huma")
state,_=env.reset()
plt.figure(figsize=(8,5))
plt.ion()
plt.show()
plot_t=[]
force=0.0
#state=np.append(state,force)
plot_x_hat=[]
plot_theta_hat=[]
state_deq = deque([np.zeros_like(state) for _ in range(10)],maxlen=10)
state_deq.append(state)
force_deq=deque([[np.zeros_like(force)] for _ in range(10)],maxlen=10)
state_init=np.array(state_deq)
model=LSTM_test()
model.load_weights('sample_model3d3')
plot_force=[]
plot_x=[]
plot_theta=[]
t=0
# Apply the MPC control

while True:
    x = np.concatenate((state_deq,force_deq),axis=1).reshape([1, 10, 5])
    cart_position_hat, cart_position_dot_hat, theta_real_hat, theta_dot_real_hat = model.predict(x)[0]
    cart_position, cart_position_dot, theta_real, theta_dot_real = env.step(force)

    u = mpc_controller(state_deq,force_deq)

    next_state = np.array([cart_position_hat, cart_position_dot_hat, theta_real_hat, theta_dot_real_hat])
    force = u
    plot_x.append(cart_position)
    plot_theta.append(theta_real)
    plot_force.append(force)
    plot_t.append(t)

    #state = np.append(next_state, force)
    state_deq.append(next_state)
    force_deq.append(u)
    t += 1
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