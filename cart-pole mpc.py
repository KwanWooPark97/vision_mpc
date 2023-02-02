import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from gekko import GEKKO
import math
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
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5  # actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0
tau = 0.05  # seconds between state updates
pi=math.pi

tau = m.Const(value=0.05)
Kp = m.Const(value=1)
Kd=10
#z_dot=x_dot
mPend=1.0
L=0.5
g=9.8
mCart=1.0
#xacc=(force - Kd*z_dot - mPend*L*theta_dot**2*math.sin(theta) + mPend*g*math.sin(theta)*math.cos(theta)) / (mCart + mPend*math.sin(theta)**2)
#thetaacc=((force - Kd*z_dot - mPend*L*theta_dot**2*math.sin(theta))*math.cos(theta)/(mCart + mPend) + g*math.sin(theta)) / (L - mPend*L*math.cos(theta)**2/(mCart + mPend))
m.F = m.MV(value=0.1)
m.x = m.CV(value=cart_position_init,lb=-4.8,ub=4.8)
m.x_dot=m.CV(value=cart_position_dot_init)
m.theta=m.CV(value=theta_init)
m.theta_dot=m.CV(value=theta_dot_init)
m.thetaacc=m.Var(value=0)
#m.temp = m.Var(value=0)
m.Obj(m.F**2)
m.Equation(m.thetaacc==((m.F - Kd*m.x_dot - mPend*L*m.theta_dot**2*m.sin(m.theta))*m.cos(m.theta)/(mCart + mPend) + g*m.sin(m.theta)) / (L - mPend*L*m.cos(m.theta)**2/(mCart + mPend)))
#m.Equation(m.temp == (m.F + polemass_length * m.theta_dot ** 2 * m.sin(m.theta*math.pi/180)) / total_mass)
m.Equation(m.x.dt()==m.x_dot)
m.Equation(m.x_dot.dt() ==(m.F - Kd*m.x_dot - mPend*L*m.theta_dot**2*m.sin(m.theta) + mPend*g*m.sin(m.theta)*m.cos(m.theta)) / (mCart + mPend*m.sin(m.theta)**2))
m.Equation(m.theta.dt()==m.theta_dot)
m.Equation(m.theta_dot.dt()==m.thetaacc)
'''
m.Equation(m.thetaacc==gravity * m.sin(m.theta*math.pi/180) - m.cos(m.theta*math.pi/180) * m.temp) / (length * (4.0 / 3.0 - masspole * m.cos(m.theta*math.pi/180) ** 2 / total_mass))
m.Equation(m.temp == (m.F + polemass_length * m.theta_dot ** 2 * m.sin(m.theta*math.pi/180)) / total_mass)
m.Equation(m.x.dt()==m.x_dot)
m.Equation(m.x_dot.dt() == m.temp - polemass_length * m.thetaacc * m.cos(m.theta*math.pi/180) / total_mass)
m.Equation(m.theta.dt()==m.theta_dot)
m.Equation(m.theta_dot.dt()==m.thetaacc)'''
#MV tuning
m.F.STATUS = 1
m.F.FSTATUS = 0
#m.F.DMAX = 10
#m.F.DMAXHI = 10   # constrain movement up
#m.F.DMAXLO = -10 # quick action down

#CV tuning
m.x.STATUS = 1
m.x.FSTATUS = 1
m.x.TR_INIT = 0
m.x.TAU = 0.05
m.x.SP=0

m.x_dot.STATUS = 1
m.x_dot.FSTATUS = 0
m.x_dot.TR_INIT = 0
m.x_dot.TAU = 0.05

m.theta.STATUS = 1
m.theta.FSTATUS = 1
m.theta.TR_INIT = 0
m.theta.TAU = 0.05
m.theta.SP=0

m.theta_dot.STATUS = 1
m.theta_dot.FSTATUS = 0
m.theta_dot.TR_INIT = 0
m.theta_dot.TAU = 0.05

DT = 0.02 # deadband

m.options.CV_TYPE = 2
m.options.IMODE = 6
#m.options.SOLVER = 3

#%% define CSTR model
def pendulum(state,t,u):
    # Inputs (1):
    # Force
    force = u

    # States (4):
    #x = state[0]
    #x_dot=state[1]
    theta=state[2]
    theta_dot=state[3]

    costheta = math.cos(math.radians(theta))
    sintheta = math.sin(math.radians(theta))
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5  # actually half the pole's length
    polemass_length = masspole * length
    # For the interested reader:
    # https://coneural.org/florian/papers/05_cart_pole.pdf
    temp = ( force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    # Return xdot:
    xdot = np.zeros(4)
    xdot[3]=thetaacc
    xdot[2]=xdot[3]
    xdot[1] = xacc
    xdot[0] = xdot[1]
    return xdot

# Time Interval (min)
t = np.linspace(0,8,401)
# Store results for plotting
cart_position = np.zeros(len(t))
cart_position_dot = np.zeros(len(t))
theta = np.zeros(len(t))
theta_dot = np.zeros(len(t))
force=np.zeros(len(t))

x_sp=np.zeros(len(t))
theta_sp=np.zeros(len(t))
# Set point steps
x_sp[0:100] = 0
x_sp[100:200] = 0
x_sp[200:300] = 0
x_sp[300:] = 0

# Create plot
plt.figure(figsize=(10,7))
plt.ion()
plt.show()

# Simulate CSTR
for i in range(len(t)-1):
    # simulate one time period (0.05 sec each loop)
    ts = [t[i],t[i+1]]
    y = odeint(pendulum,x0,ts,args=(force[i],))
    # retrieve measurements
    cart_position[i+1] = y[-1][0]
    cart_position_dot[i+1] = y[-1][1]
    theta[i+1] = y[-1][2]
    theta_dot[i+1] = y[-1][3]
    # insert measurement
    m.x.MEAS = cart_position[i+1]
    m.theta.MEAS = theta[i+1]
    # solve MPC
    m.solve(disp=False)
    print(theta[i])
    m.x.SPHI = x_sp[i+1] + DT
    m.x.SPLO = x_sp[i+1] - DT
    m.theta.SPHI=theta_sp[i+1] +DT
    m.theta.SPLO=theta_sp[i+1] -DT

    # retrieve new Tc value
    force[i+1] = m.F.NEWVAL
    # update initial conditions
    x0[0] = cart_position[i+1]
    x0[1] = cart_position_dot[i+1]
    x0[2] = theta[i+1]
    x0[3] = theta_dot[i+1]

    #%% Plot the results
    plt.clf()
    plt.subplot(3,1,1)
    plt.plot(t[0:i],force[0:i],'b--',lw=3)
    plt.ylabel('input Force')
    plt.legend(['force'],loc='best')

    plt.subplot(3,1,2)
    plt.plot(t[0:i],cart_position[0:i],'b.-',lw=3,label=r'$position$')
    plt.plot(t[0:i], x_sp[0:i], 'r-', lw=3, label=r'$position_{sp}$')
    plt.ylabel(r'cart position')
    plt.legend(loc='best')

    plt.subplot(3,1,3)
    plt.plot(t[0:i],theta_sp[0:i],'r-',lw=3,label=r'$theta_{sp}$')
    plt.plot(t[0:i],theta[0:i],'b.-',lw=3,label=r'$theta_{meas}$')
    plt.ylabel('theta')
    plt.xlabel('Time (min)')
    plt.legend(loc='best')
    plt.draw()
    plt.pause(0.01)

plt.show()