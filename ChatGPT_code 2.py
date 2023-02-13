import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from collections import deque
from cartpole_gym_env import CartPoleEnv
import matplotlib.pyplot as plt
import scipy


class LSTM_test(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.s1 = LSTM(512, input_shape=(None,10,5),return_sequences=True)
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
# Define the plant model
plant = LSTM_test()

# Define the control horizon and the prediction horizon
control_horizon = 10
prediction_horizon = 20

# Define the state and control bounds
state_bounds = np.array([[-100, 100], [-100, 100], [-3*np.pi, 3*np.pi], [-100, 100]])
control_bounds = np.array([[-30, 30]])

# Define the initial state and setpoint
initial_state = np.array([[0, 0, -np.pi, 0]])
setpoint = np.array([[0, 0, 0, 0]])

def mpc_controller(initial_state, setpoint, control_horizon, prediction_horizon, state_bounds, control_bounds):
    # Define the cost function
    def cost_function(state, control):
        state_error = state - setpoint
        control_error = control
        cost = np.sum(state_error**2) + np.sum(control_error**2)
        return cost

    # Define the MPC loop
    state = initial_state
    for i in range(control_horizon):
        # Define the optimization problem
        def optimization_problem(control):
            cost = 0
            state = initial_state
            for j in range(prediction_horizon):
                next_state = plant(np.concatenate((state, np.expand_dims(control, axis=1)), axis=-1))
                state = next_state
                cost += cost_function(state, control)
            return cost

        # Solve the optimization problem
        control = scipy.optimize.minimize(optimization_problem, control, bounds=control_bounds)

        # Apply the control
        next_state = plant(np.concatenate((state, np.expand_dims(control, axis=1)), axis=-1))
        state = next_state

    return control

# Call the MPC controller
control = mpc_controller(initial_state, setpoint, control_horizon, prediction_horizon, state_bounds, control_bounds)

import numpy as np
import tensorflow as tf


class Plant(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm1 = tf.keras.layers.LSTM(512, input_shape=(None, 10, 5), return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(256, return_sequences=True)
        self.lstm3 = tf.keras.layers.LSTM(128)
        self.dense = tf.keras.layers.Dense(4)

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.lstm2(x)
        x = self.lstm3(x)
        x = self.dense(x)

        return x


plant = Plant()
plant.load_weights('sample_model3d3')

state_buffer = []
T = 100  # Number of time steps
N = 10  # Number of prediction steps for MPC

for t in range(T):
    # Keep track of past 10 time steps of state in a buffer
    state_buffer.append(state)
    if len(state_buffer) > 10:
        state_buffer.pop(0)

    # Concatenate the latest state with the previous 9 time steps
    input_data = np.concatenate(state_buffer + [np.expand_dims(control, axis=0)], axis=1)

    # Reshape the input data to match the model's expected input shape
    input_data = np.expand_dims(input_data, axis=0)
    input_data = np.expand_dims(input_data, axis=0)

    # Use MPC to control the system
    predicted_states = []
    for i in range(N):
        # Get the next state from the model
        next_state = plant(input_data)
        predicted_states.append(next_state)

        # Update the input data for the next prediction
        input_data = np.concatenate(state_buffer + [next_state, np.expand_dims(control, axis=0)], axis=1)
        input_data = np.expand_dims(input_data, axis=0)
        input_data = np.expand_dims(input_data, axis=0)

    # Choose the control signal that minimizes the cost function
    cost = ...  # Define your cost function here
    best_control = ...  # Choose the best control signal based on the cost function

    # Apply the chosen control signal to the system
    control = best_control
