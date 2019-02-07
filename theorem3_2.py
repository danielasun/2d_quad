import numpy as np
from numpy import pi, sin, cos
import math
import matplotlib.pyplot as plt

M = np.block([[np.zeros((3, 3)), np.eye(3)], [np.zeros((3, 3)), np.zeros((3, 3))]])

r = .127  # 5 inches => meters
m = .1  # kg
Fmin = np.r_[0, 0]
Fmax = np.r_[10, 10]
L = 1
mu = 2364 # taking F1, F2 to be a constant 5


def B(theta):
    return 1 / m * np.array([[0, 0],
                             [0, 0],
                             [0, 0],
                             [-np.sin(theta), -np.sin(theta)],
                             [np.cos(theta), np.cos(theta)],
                             [3 / r, -3 / r]]
                            )

G = np.r_[0, 0, 0, 0, -9.8, 0]

def xprime(M, B, G, x, u):
    """

    :param M: 6x6 mass matrix
    :param B: 6x2  the matrix for input
    :param G: 6x1 matrix of gravity
    :param x: 6x1 current state of system
    :param u: 2x1 input effort
    :return: x', the rate of change of the system
    """

    return np.dot(M, x) + np.dot(B(x[2]), u) + G

ic = np.r_[0,5,pi/10,5,5,0]
ic2 = np.r_[.1,5,pi/10,5,5,0]
u0 = np.r_[0,0]
print(xprime(M, B, G, ic, u0))

# simulate w/ euler's method
dt = .001
tfinal = 1
times = np.arange(0,tfinal, dt)



def desolve_int(f, ic, dt, tfinal , u_func=None):
    times = np.arange(0,tfinal, dt)
    x = ic
    xlist = np.zeros((math.floor(tfinal/dt), 6))
    ulist = np.zeros((math.floor(tfinal/dt), 2))

    for i in range(len(times)):
        xdes = np.r_[0,5,0,0,0,0]
        # u = pd_ctrl(xdes - x, (xdes - x) - err,
        #             dt = dt,
        #             kp_theta=kp_theta,
        #             kd_theta=kd_theta,
        #             kp_z=kp_z,
        #             kd_z=kd_z,
        #             kp_x=kp_x,
        #             kd_x=kd_x)

        err = xdes - x

        u =  np.clip(u_func(x), Fmin, Fmax)

        err = xdes - x
        xlist[i,:] = x
        ulist[i,:] = u
        x = f(M, B, G, x, u)*dt + x
    return xlist, ulist

xlist, ulist = desolve_int(xprime, ic, dt, tfinal, lambda x: np.r_[10,0]) # engines at full throttle
zlist, ulist2 = desolve_int(xprime, ic2, dt, tfinal, lambda x: np.r_[0,0]) # engines off
# make the plot
plt.figure()

plt.subplot(211)
for i, l in zip(range(2), ['x','z']):
    plt.plot(times, xlist[:,i], label=l)
legend = plt.legend(loc='lower right', shadow=True, fontsize='small')
plt.xlabel('time')
plt.ylabel('m/s or rad/s')
plt.title('position vs. time')

plt.subplot(212)
plt.plot(times, xlist[:,2], label='theta')
plt.xlabel('time')
plt.title('theta vs. time')

plt.figure()
plt.subplot(211)
for i, l in zip(range(3,6), ['xdot','zdot','thetadot']):
    plt.plot(times, xlist[:,i], label=l)
plt.xlabel('time')
plt.ylabel('m/s or rad/s')
plt.title('xdot, zdot, thetadot')
legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
plt.subplot(212)
for i, l in zip(range(2), ['F1','F2']):
    plt.plot(times, ulist[:,i], label=l)
plt.xlabel('time')
plt.ylabel('N')
plt.title('F1,F2')
#
show_traj = 1
if show_traj:
    plt.figure()
    plt.plot(xlist[:,0], xlist[:,1], 'b', label="thrust on")
    plt.plot(zlist[:,0], zlist[:,1], 'r--', label='thrust off')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('X-Z trajectory of quadcopters, t = [0,1]')

# calculate mu
gxnorm = np.arange(0,len(times))
for i in range(len(times)):
    gxnorm[i] = np.linalg.norm(B(xlist[i,2])@ulist[i])
mu = gxnorm.max(axis=0)
print('mu = {}'.format(mu))

plt.figure()
plt.plot(times,np.linalg.norm(xlist-zlist, axis=1), label='||x(t) - z(t)||')
plt.plot(times,np.linalg.norm(ic-ic2)*L*np.exp(times) + mu*times*np.exp(L*times), label='||x0-z0||e^(Lt) + ute^(L(t))')
legend = plt.legend(loc='upper left', shadow=True, fontsize='small')
plt.xlabel('time')
plt.ylabel('error norm')

plt.figure()
interval = int(.05//dt)
times = times[:interval]
plt.plot(times,np.linalg.norm(xlist[:interval,:]-zlist[:interval,:], axis=1), label='||x(t) - z(t)||')
plt.plot(times,np.linalg.norm(ic-ic2)*L*np.exp(times) + mu*times*np.exp(L*times), label='||x0-z0||e^(Lt) + ute^(L(t))')
legend = plt.legend(loc='upper left', shadow=True, fontsize='small')
plt.xlabel('time')
plt.ylabel('error norm')

plt.show()
