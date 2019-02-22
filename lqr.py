import numpy as np
from numpy import pi, sin, cos
import math
import matplotlib.pyplot as plt
import sympy as sym

M = np.block([[np.zeros((3,3)), np.eye(3)],[np.zeros((3,3)), -1e-6*np.ones((3,3))]])

r = .127 # 5 inches => meters
m = .1 # kg
g = 9.8
Fmin = np.r_[0,0]
Fmax = np.r_[10,10]
K = np.array([[-0.70710678,  0.70710678,  3.86763713, -1.02865366,  0.7554539,  0.71859098],
              [ 0.70710678,  0.70710678, -3.86763713,  1.02865366,  0.7554539,  -0.71859098]])

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

ic = np.r_[.1,2,1.6,0,0,0]
u0 = np.r_[0,0]
print(xprime(M, B, G, ic, u0))

# euler's method
dt = .001
tfinal = 5
times = np.arange(0,tfinal, dt)

# simulate
x = ic
err = np.r_[0,0,0,0,0,0]
xdes = np.r_[0,0,0,0,0,0]
xlist = np.zeros((math.floor(tfinal/dt), 6))
ulist = np.zeros((math.floor(tfinal/dt), 2))

show_traj = 1
show_traj = 0

for i in range(len(times)):
    u =  np.clip(-K@x + np.r_[.5*m*g/cos(x[2]),.5*m*g/cos(x[2])], Fmin, Fmax)

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
plt.title('velocity vs. time')
legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
plt.subplot(414)
for i, l in zip(range(2), ['F1','F2']):
    plt.plot(times, ulist[:,i], label=l)
plt.title('F1, F2 vs. time')
legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
#
if show_traj:
    plt.figure(2)
    plt.plot(xlist[:,0], xlist[:,1], 'b.')
    plt.xlabel('x')
    plt.ylabel('z')

plt.show()
