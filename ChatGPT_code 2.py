import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from collections import deque
from cartpole_gym_env import CartPoleEnv
import matplotlib.pyplot as plt
from scipy import optimize

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

# Define the plant model
plant = LSTM_test()
plant.load_weights('sample_model3d2')
# Define the control horizon and the prediction horizon
control_horizon = 3
prediction_horizon = 3

# Define the state and control bounds
state_bounds = np.array([[-100, 100], [-100, 100], [-3*np.pi, 3*np.pi], [-100, 100]])
control_bounds = (-30,30)
bnd=((-30.0,30.0),)
# Define the initial state and setpoint
initial_state = np.array([[0, 0, -np.pi, 0]])
setpoint = np.array([[0, 0, 0, 0]])

def mpc_controller(initial_state, initial_force,setpoint, control_horizon, prediction_horizon, state_bounds, bnd):
    # Define the cost function
    def cost_function(state, control):
        state_error = np.array(state)
        control_error = control
        cost = 0.5*state_error[0]**2+0.1*state_error[1]**2+2*state_error[2]**2+0.1*state_error[3]**2 + 0.1*control_error**2
        return cost
    control=initial_force[-1]
    # Define the MPC loop
    force = deque(np.array(initial_force),maxlen=10)
    state = deque(np.array(initial_state),maxlen=10)

        # Define the optimization problem
    def optimization_problem(control):
        cost = 0
        state_buffer = deque(np.array(state),maxlen=10)
        force_buffer= deque(np.array(force),maxlen=10)
        control=np.reshape(control,(10,1))
        for j in range(prediction_horizon):
            force_buffer.append(control[j])

            input_data=np.concatenate(((state_buffer, force_buffer)), axis=1).reshape([1,10,5])
            next_state = plant(input_data)[0]
            state_buffer.append(next_state)
            cost += cost_function(next_state, control[j][0])

        return cost
    def constraint_function(u):
        return 1 - np.linalg.norm(u, ord=np.inf)

    con = {'type': 'ineq', 'fun': constraint_function}
    # Solve the optimization problem
    result = optimize.minimize(optimization_problem, force, bounds=((-30,30),(-30,30),(-30,30),(-30,30),(-30,30),(-30,30),(-30,30),(-30,30),(-30,30),(-30,30),))
    '''force.append(result.x)
    # Apply the control
    next_state = plant(np.concatenate((state, force), axis=1).reshape([1,10,5]))[0]
    state.append(next_state)'''

    return result.x[0]

# Call the MPC controller
env=CartPoleEnv("huma")
state,_=env.reset()
plt.figure(figsize=(8,5))
plt.ion()
plt.show()
plot_t=[]
force=np.array([0.0])

#state=np.append(state,force)
plot_x_hat=[]
plot_theta_hat=[]
state_deq = deque([np.zeros_like(state) for _ in range(10)],maxlen=10)
state_deq.append(state)
state_deq_mpc = deque([np.zeros_like(state) for _ in range(10)],maxlen=10)
state_deq_mpc.append(state)
force_deq=deque([np.zeros_like(force) for _ in range(10)],maxlen=10)
force_deq_mpc=deque([np.zeros_like(force) for _ in range(10)],maxlen=10)

plot_force=[]
plot_x=[]
plot_theta=[]
t=0
# Apply the MPC control

while True:
    state_deq_mpc=state_deq.copy()
    force_deq_mpc=force_deq.copy()
    control = mpc_controller(state_deq_mpc, force_deq_mpc, setpoint, control_horizon, prediction_horizon, state_bounds, bnd)
    force = control
    force= np.reshape(force,(1,))
    force_deq.append(force)
    x = np.concatenate((state_deq,force_deq),axis=1).reshape([1, 10, 5])
    cart_position_hat, cart_position_dot_hat, theta_real_hat, theta_dot_real_hat = plant.predict(x)[0]

    cart_position, cart_position_dot, theta_real, theta_dot_real = env.step(force[0])



    next_state = np.array([cart_position, cart_position_dot, theta_real, theta_dot_real])



    #state = np.append(next_state, force)
    state_deq.append(next_state)

    plot_x.append(cart_position)
    plot_x_hat.append(cart_position_hat)
    plot_theta.append(theta_real)
    plot_theta_hat.append(theta_real_hat)
    plot_force.append(force)
    plot_t.append(t)
    t += 1

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


