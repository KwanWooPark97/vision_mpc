#import tensorflow as tf
import numpy as np
import math
import cvxpy as cp
import matplotlib.pyplot as plt

def plant(y,t,u):
    #input u= w1  output cbt,ht
    w2, cb1, cb2, k1, k2 =0.1 , 24.9, 0.1, 1.0,1.0
    dhdt=u+w2-0.2*math.sqrt(y[0])
    dcbdt=(cb1-y[1])*(u/y[0])+(cb2-y[1])*(w2/y[0])-(k1*y[1])/((1+k2*y[1])**2)

    output=[dhdt,dcbdt]

    return output

def mpc_model(x_ref):
    n = 2
    m = 2
    T = 20

    A =np.array([-5.0,-0.3427,47.68,2.785]).reshape(2,2)
    B = np.array([0.0,1.0,0.3,0.0]).reshape(2,2)
    x = cp.Variable((n, T + 1))
    u = cp.Variable((m, T))
    x_0=np.array([8.5698,311.2639])
    cost = 0
    constr = []

    for t in range(T):
        cost += cp.sum_squares(x_ref-x[:, t+1]) + cp.sum_squares(u[:, t])
        constr += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t]]

    # sums problem objectives and concatenates constraints.
    constr += [x[:, 0] == x_0]
    constr += [x[0,:] <= 10.0]
    constr += [x[0, :] >= 0.0]
    constr += [x[1, :] <= 390.0]
    constr += [x[1, :] >= 310.0]
    problem = cp.Problem(cp.Minimize(cost), constr)
    problem.solve()
    print(problem.status)
    print(x.value)
    print(u.value)

    rx0=np.array(x.value[0, :]).flatten()
    rx1=np.array(x.value[1,:]).flatten()
    u1=np.array(u.value[0,:]).flatten()
    u2=np.array(u.value[1,:]).flatten()
    plt.subplot(2,2,1)
    plt.title('state_0')
    plt.plot(rx0)
    plt.subplot(2,2,2)
    plt.title('state_1')
    plt.plot(rx1)
    plt.subplot(2, 2, 3)
    plt.title('input_0')
    plt.plot(u1)
    plt.subplot(2, 2, 4)
    plt.title('input_1')
    plt.plot(u2)

    plt.show()







if __name__ == '__main__':
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
    mpc_model(np.array([8.0,350.0]))