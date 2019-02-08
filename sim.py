from typing import Optional, Any

import numpy as np
from numpy import pi, sin, cos
import math
import matplotlib.pyplot as plt


M = np.block([[np.zeros((3,3)), np.eye(3)],[np.zeros((3,3)), np.zeros((3,3))]])

r = .127 # 5 inches => meters
m = .1 # kg
Fmin = np.r_[0,0]
Fmax = np.r_[10,10]

def B(theta):
    return 1/m*np.array([[0, 0],
                     [0, 0],
                     [0, 0],
                     [-np.sin(theta), -np.sin(theta)],
                     [np.cos(theta), np.cos(theta)],
                     [3/r, -3/r]]
             )

G = np.r_[0,0,0,0,-9.8,0]

def xprime(M, B, G, x, u):
    """

    :param M: 6x6 mass matrix
    :param B: 6x2 matrix for input
    :param G: 6x1 matrix of gravity
    :param x: 6x1 current state of system
    :param u: 2x1 input effort
    :return: x', the rate of change of the system
    """

    return np.dot(M,x) + np.dot(B(x[2]), u) + G

def pid_ctrl(err, derr, int_err, dt, kp, kd, ki):
    """
    PID controller

    :param err: error in the state (desired - current)
    :param derr: change in the error in the state
    :param int_err: sum of error in the state
    :param dt: time step
    :param kp: proportional gain
    :param kd: derivative gain
    :param ki: integral gain
    :return: controller output
    """

    return kp*err + kd/dt*derr + ki*int_err

class PID_Controller(object):
    def __init__(self, dt, P=0.0, I=0.0, D=0.0, reset_tolerance=0.01, windup_max=None,):
        """

        :param dt: timestep
        :param P: proportional gain
        :param I: integral gain
        :param D: derivative gain
        :param reset_tolerance: how much the setpoint has to change in order to trigger resetting the error integral
        :param windup_max: maximum value of the error integral
        """
        self.P = P
        self.D = D
        self.I = I

        self.prev_setpoint = 0
        self.prev_error = 0
        self.error_integral = 0
        self.reset_tolerance = reset_tolerance
        self.dt = dt
        self.windup_max = windup_max

    def calc_feedback(self, setpoint, measurement):
        error = setpoint-measurement

        # P
        p_term = np.dot(self.P, setpoint - measurement)

        # D
        d_term = self.D/self.dt * (error - self.prev_error)

        # I
        # reset error integral if setpoint changes
        if abs(setpoint - self.prev_setpoint) > self.reset_tolerance:
            print("reset tolerance: {}".format(self))
            self.error_integral = 0
        self.error_integral += error*dt

        # catch windup issues
        if self.windup_max is not None: # if we are using integral windup
            if self.error_integral > self.windup_max:
                self.error_integral = self.windup_max
            elif self.error_integral < -self.windup_max:
                self.error_integral = -self.windup_max

        i_term = np.dot(self.I, self.error_integral)

        self.prev_error = error
        self.prev_setpoint = setpoint
        # print("p: {}, d: {}, int: {}".format(p_term, d_term, self.error_integral))
        return p_term + d_term + i_term

def pd_ctrl(err, derr, dt, kp_theta, kd_theta, kp_z, kd_z, kp_x, kd_x):
    """
    Propoprtional control over error in state.
    theta is counter clockwise positive

    :param kp_theta:
    :param kp_z:
    :param err: error in x,z,theta,xdot, zdot and thetadot
    :param derr: change in err between last time and this time
    :return: 2x1 vector (F1,F2) with thrusts for the quadcopter
    """

    theta_stab = np.r_[kp_theta*err[2], -kp_theta*err[2]] + 1/dt*np.r_[kd_theta*derr[2], -kd_theta*derr[2]]
    zstab = np.r_[kp_z*err[4] + kp_z*err[4]]
    xstab = np.r_[kp_x*err[0], -kp_x*err[0]]
    cmd = theta_stab + zstab + xstab

    cmd = np.clip(cmd, Fmin, Fmax)

    return cmd

ic = np.r_[0,5,pi/10,.1,0,0]
u0 = np.r_[0,0]
print(xprime(M, B, G, ic, u0))

# euler's method
dt = .001
tfinal = 10
times = np.arange(0,tfinal, dt)

# simulate
x = ic
err = np.r_[0,0,0,0,0,0]
xdes = np.r_[0,5,0,0,0,0]
xlist = np.zeros((math.floor(tfinal/dt), 6))
ulist = np.zeros((math.floor(tfinal/dt), 2))

theta_ctrl = PID_Controller(dt, P=.5, I=.001, D=.05)
z_ctrl = PID_Controller(dt, P=.5, I=.03, D=.1)
x_ctrl = PID_Controller(dt, P=0.4, I=.05, D=.05)
show_traj = 1
show_traj = 0

for i in range(len(times)):

    if i > len(times)/3:
        xdes = np.r_[.5,5,0,0,0,0]

    theta_des = x_ctrl.calc_feedback(xdes[0], x[0])  # type: float
    xdes[2] = -theta_des
    err = xdes - x
    theta_u = theta_ctrl.calc_feedback(xdes[2], x[2])
    z_u = z_ctrl.calc_feedback(xdes[1], x[1])

    u =  np.clip(np.r_[theta_u +z_u,
                       -theta_u + z_u], Fmin, Fmax)

    err = xdes - x
    xlist[i,:] = x
    ulist[i,:] = u
    x = xprime(M, B, G, x, u)*dt + x

# make the plot
plt.figure(1)
plt.subplot(411)
for i, l in zip(range(2), ['x','z']):
    plt.plot(times, xlist[:,i], label=l)
legend = plt.legend(loc='lower right', shadow=True, fontsize='small')
plt.xlabel('time')
plt.title('position vs. time')
plt.subplot(412)
plt.plot(times, xlist[:,2], label='theta')
plt.xlabel('time')
plt.title('theta vs. time')
plt.subplot(413)
for i, l in zip(range(3,6), ['xdot','zdot','thetadot']):
    plt.plot(times, xlist[:,i], label=l)
plt.xlabel('time')
legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
plt.subplot(414)
for i, l in zip(range(2), ['F1','F2']):
    plt.plot(times, ulist[:,i], label=l)
#
if show_traj:
    plt.figure(2)
    plt.plot(xlist[:,0], xlist[:,1], 'b.')
    plt.xlabel('x')
    plt.ylabel('z')

plt.show()
