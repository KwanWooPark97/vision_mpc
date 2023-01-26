import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gekko import GEKKO
import math
import datetime
from cartpole_gym_env import CartPoleEnv
# 팬듈럼 생각해보면 입력은(MV) F 상태 변수는(CV) 4개 cart_position,cart_position_dot,theta,theta_dot

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
x0[2]=theta_init
x0[3]=theta_dot_init

#%% GEKKO nonlinear MPC
m = GEKKO(remote=False)

#m.time = [0,0.05,0.1,0.15]
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5  # actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0
tau = 0.08  # seconds between state updates
pi=math.pi

tau = m.Const(value=0.08)
Kp = m.Const(value=1)

F=m.Var(value=0)
theta = m.Var(value=0)
theta_dot = m.Var(value=0)
x = m.Var(value=-1.0)
x_dot = m.Var(value=0)

thetaacc=m.Var(value=0)
temp = m.Var(value=0)

m.Equation(thetaacc==gravity * m.sin(theta*math.pi/180) - m.cos(theta*math.pi/180) * temp) / (length * (4.0 / 3.0 - masspole * m.cos(theta*math.pi/180) ** 2 / total_mass))
m.Equation(temp == (F + polemass_length * theta_dot ** 2 * m.sin(theta*math.pi/180)) / total_mass)
m.Equation(x.dt()==x_dot)
m.Equation(x_dot.dt() == temp - polemass_length * thetaacc * m.cos(theta*math.pi/180) / total_mass)
m.Equation(theta.dt()==theta_dot)
m.Equation(theta_dot.dt()==thetaacc)

m.Obj(x**2)
m.Obj(x_dot**2)
m.Obj(theta**2)
m.Obj(theta_dot**2)

m.Obj(F**2)

m.options.IMODE = 6

# Set point steps
force=0.0
# Simulate CSTR
plot_force=[]
plot_x=[]
plot_theta=[]
t=[]
env=CartPoleEnv("human")
_,_=env.reset()
plt.figure(figsize=(8,5))
plt.ion()
plt.show()
plot_t=[]
t=0
times=0
while True:
    m.time = [times,times+0.08,times+0.16,times+0.24,times+0.32,times+0.4]
    #start_time = datetime.datetime.now()
    #env.render()
    t+=1
    # simulate one time period (0.05 sec each loop)
    #ts = [t,next_t]
    #y = odeint(pendulum,x0,ts,args=(force[i],))
    print(force)
    cart_position_dt, cart_position_dot_dt, theta_dt, theta_dot_dt = env.step(force)

    # retrieve measurements
    # insert measurement
    x = cart_position_dt
    x_dot=cart_position_dot_dt
    theta = theta_dt
    theta_dot=theta_dot_dt
    # solve MPC


    m.solve(disp=False)

    # retrieve new Tc value
    force =F.value[0]
    plot_x.append(cart_position_dt)
    plot_theta.append(theta)
    plot_force.append(force)
    plot_t.append(t)
    times+=0.08
    '''end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print("5",elapsed_time.seconds)'''
    # update initial conditions
    #x0[0] = cart_position
    #x0[1] = cart_position_dot
    #x0[2] = theta
    #x0[3] = theta_dot
    plt.clf()
    plt.subplot(3, 1, 1)
    plt.plot(plot_t, plot_force, 'b--', lw=3)
    plt.ylabel('input Force')
    plt.legend(['force'], loc='best')

    plt.subplot(3, 1, 2)
    plt.plot(plot_t, plot_x, 'b.-', lw=3, label=r'$position$')
    #plt.plot(plot_t, np.zeros(plot_t), 'r-', lw=3, label=r'$position_{sp}$')
    plt.axhline(0.0, 0.1, 0.9, color='r', linestyle='-', label=r'$position_{sp}$')
    plt.ylabel(r'cart position')
    plt.legend(loc='best')

    plt.subplot(3, 1, 3)
    #plt.plot(plot_t, np.zeros(plot_t), 'r-', lw=3, label=r'$theta_{sp}$')
    plt.axhline(0.0, 0.1, 0.9, color='r', linestyle='-', label=r'$position_{sp}$')
    plt.plot(plot_t, plot_theta, 'b.-', lw=3, label=r'$theta_{meas}$')
    plt.ylabel('theta')
    plt.xlabel('Time (min)')
    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.01)
