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
m.time = [0,0.02,0.04,0.06,0.08,0.1,0.12,0.15,0.2]
#m.time = [0,0.05,0.1,0.15]
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5  # actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0
tau = 0.012  # seconds between state updates
pi=math.pi

tau = m.Const(value=0.011)
Kp = m.Const(value=1)

m.F = m.MV(value=-1.0)
m.x = m.CV(value=cart_position_init,lb=-4.8,ub=4.8)
m.x_dot=m.CV(value=cart_position_dot_init)
m.theta=m.CV(value=theta_init)
m.theta_dot=m.CV(value=theta_dot_init)
m.thetaacc=m.Var(value=0)
m.temp = m.Var(value=0)
m.Equation(m.thetaacc==gravity * m.sin(m.theta*math.pi/180) - m.cos(m.theta*math.pi/180) * m.temp) / (length * (4.0 / 3.0 - masspole * m.cos(m.theta*math.pi/180) ** 2 / total_mass))
m.Equation(m.temp == (m.F + polemass_length * m.theta_dot ** 2 * m.sin(m.theta*math.pi/180)) / total_mass)
m.Equation(m.x.dt()==m.x_dot)
m.Equation(m.x_dot.dt() == m.temp - polemass_length * m.thetaacc * m.cos(m.theta*math.pi/180) / total_mass)
m.Equation(m.theta.dt()==m.theta_dot)
m.Equation(m.theta_dot.dt()==m.thetaacc)
#MV tuning
m.F.STATUS = 1
m.F.FSTATUS = 0
m.F.DMAX = 10
#m.F.DMAXHI = 10   # constrain movement up
#m.F.DMAXLO = -10 # quick action down

m.Obj(m.x**2)
m.Obj(m.x_dot**2)
m.Obj(m.theta**2)
m.Obj(m.theta_dot**2)

#CV tuning
m.x.STATUS = 1
m.x.FSTATUS = 1
m.x.TR_INIT = 0
m.x.TAU = 0.011

m.x_dot.STATUS = 1
m.x_dot.FSTATUS = 0
m.x_dot.TR_INIT = 0
m.x_dot.TAU = 0.011

m.theta.STATUS = 1
m.theta.FSTATUS = 1
m.theta.TR_INIT = 0
m.theta.TAU = 0.011

m.theta_dot.STATUS = 1
m.theta_dot.FSTATUS = 0
m.theta_dot.TR_INIT = 0
m.theta_dot.TAU = 0.011

DT = 0.01 # deadband

#m.options.CV_TYPE = 2
m.options.IMODE = 6
#m.options.SOLVER = 3

# Set point steps
m.x.SPHI = 0 + 0.5
m.x.SPLO = 0 - 0.5
m.theta.SPHI=0 +DT
m.theta.SPLO=0 -DT
force=-12.0
# Simulate CSTR
plot_force=[]
plot_x=[]
plot_theta=[]
t=[]
env=CartPoleEnv("human")
_,_=env.reset()
plt.figure(figsize=(10,7))
plt.ion()
plt.show()
plot_t=[]
t=0
while True:
    start_time = datetime.datetime.now()
    #env.render()
    t+=1
    # simulate one time period (0.05 sec each loop)
    #ts = [t,next_t]
    #y = odeint(pendulum,x0,ts,args=(force[i],))
    cart_position, cart_position_dot, theta, theta_dot = env.step(force)

    # retrieve measurements
    # insert measurement
    m.x.MEAS = cart_position
    m.theta.MEAS = theta
    # solve MPC

    m.solve(disp=False)

    # retrieve new Tc value
    force =m.F.NEWVAL
    plot_x.append(cart_position)
    plot_theta.append(theta)
    plot_force.append(force)
    plot_t.append(t)
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
