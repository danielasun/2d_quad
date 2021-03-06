# Daniel Sun 15 Mar 2019 
# nonlineardynamics

import numpy as np
from numpy import pi, sin, cos
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import sympy as sp
from sympy.interactive.printing import init_printing

init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import symbols, pprint, lambdify
import scipy.linalg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from csf import calc_csf
from safety_fn_position import gen_safety_coeffs_fn
from latex_util import round_expr, lp, lpr

def lqr(A, B, Q, R):
    """
        Solve the continuous time lqr controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u

        :param A: state matrix
        :param B: input matrix
        :param Q: state cost
        :param R: input cost
        :return: K, X, eigvals
        """

    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals


##################
# MODE SELECTION #
##################

mode_string = \
"""
'Use csf? (Y/y) for yes, anything else for no'
"""
choice_string = input(mode_string).lower()
if choice_string== 'y':
    USE_CSF = True
else:
    USE_CSF = False

mode_string = \
"""
Save animation? (Y/y) for yes, anything else for no
"""
choice_string = input(mode_string).lower()
if choice_string == 'y':
    SAVE_ANIMATION = True
else:
    SAVE_ANIMATION = False

mode_string = \
"""
Choose flight plan: (C/c) for circle, (S/s) for straight flight
"""
choice_string = input(mode_string).lower()
if choice_string == 'c':
    FLIGHT_PLAN = "CIRCLE"
elif choice_string == 's':
    FLIGHT_PLAN = "STRAIGHT"
else:
    print("Invalid choice, exiting.")
    exit(1)



# write down dynamics
ox, oz, ot, m, r, F1, F2, grav, v1, v2 = symbols('ox, oz, ot, m, r, F1, F2, grav, v1, v2')
kx, kz, kt, kxd, kzd, ktd = symbols('kx, kz, kt, kxd, kzd, ktd')
x, z, th, xd, zd, thd = symbols('x z th xd zd thd')
symbols('x z th xd zd thd')

symbol_names = {ox: '\sigma_x',
                oz: '\sigma_z',
                ot: '\sigma_\\theta',
                xd: '\dot{x}', zd: '\dot{z}', th: '\\theta', thd: '\dot{\\theta}'}

for k, ending in zip([kx, kz, kt, kxd, kzd, ktd, ], ['x', 'z', 't', '\dot{x}', '\dot{z}', '\dot{\\theta}']):
    symbol_names[k] = "k_{" + ending + "}"

# Define dynamical system
subs_dict = {ox: 1e-6, oz: 1e-6, ot: 1e-6, r: .127, m: .1, grav: 9.8}

# vectors of variables
q = sp.Matrix([x, z, th])
qd = sp.Matrix([xd, zd, thd])
u = sp.Matrix([F1, F2])
X = sp.Matrix([x, z, th, xd, zd, thd])

# matrices
J = m / (12) * (2 * r) ** 2
D = sp.Matrix(np.diag([m, m, J]))
B = sp.Matrix([[-sp.sin(th), 0],
               [sp.cos(th), 0],
               [0, 1]])
H = sp.Matrix([0, m * grav, 0])

Dinv = sp.Matrix(np.diag([1 / m, 1 / m, 1 / J]))
qdd = -Dinv @ H + Dinv @ B @ u

f = sp.Matrix([qd, -Dinv @ H]).subs(subs_dict)
g = sp.Matrix([[0, 0], [0, 0], [0, 0], Dinv @ B])

# create change equations
change_eqs = sp.lambdify((X, u), (f + g @ u).subs(subs_dict))


fm_to_u1u2 = sp.Matrix([[1/2, 1/(2*r)],
                        [1/2, -1/(2*r)]]).subs(subs_dict)

J_ = J.subs(subs_dict)
grav_ = grav.subs(subs_dict)
m_ = m.subs(subs_dict)

A_ = np.array([[0,0,1,0],
               [0,0,0,1],
               [0,0,0,0],
               [0,0,0,0]])
B_ = np.array([[0,0],[0,0],[1,0],[0,1]])

Q = np.eye(4)
R = np.eye(2)*100

K, X, eigvals = lqr(A_, B_, Q, R)




# make the plot
dt = .005

#################
# circle_flight #
#################
if FLIGHT_PLAN == "CIRCLE":
    ic = np.r_[0, 0, -.5, 0, 0, 0]
    tfinal = 10
    times = np.arange(0, tfinal, dt)
    xtraj = np.vstack([np.cos(2*np.pi*times/tfinal),
                       np.sin(2*np.pi*times/tfinal),
                       np.zeros_like(times),
                       -np.sin(2*np.pi*times/tfinal),
                       np.cos(2 * np.pi * times / tfinal),
                       np.zeros_like(times)]).T
    danger_x = -1.5
    danger_z = 0
    csf_fn = gen_safety_coeffs_fn(x0=danger_x, z0=danger_z, alpha=2)

#################
# Straight Line #
#################
if FLIGHT_PLAN == 'STRAIGHT':
    tfinal = 6
    ic = np.r_[0, .1, 0, 0, 0, 0]
    X_final = np.r_[2,0,0,0,0]
    times = np.arange(0, tfinal, dt)
    xtraj = np.vstack([np.linspace(ic[0], X_final[0],len(times)),
                       np.linspace(ic[1], X_final[1], len(times)),
                       np.zeros_like(times),
                       1/dt*np.diff(np.linspace(ic[0], X_final[0], len(times)), prepend=0),
                       1/dt*np.diff(np.linspace(ic[1], X_final[1], len(times)), prepend=0),
                       np.zeros_like(times)]).T
    danger_x = .8
    danger_z = 0
    csf_fn = gen_safety_coeffs_fn(x0=danger_x, z0=danger_z, alpha=.5)



# controller
def hierarchical_ctrl(X, Xdes, xdd_ref=None, zdd_ref=None):
    """
    State dependent control that controls the quadcopter in x and z by choosing thrust to cancel out gravity and rapidly
    servoing the angle to achieve the desired acceleration in x.

    :param X: 6 vec current states [x z theta xd zd thetad]
    :param Xdes: # 6 vec desired state value for X
    :return: 2x1 control output vector [T,M]
    """

    # Control parameters:
    wn_x = 1
    xi_x = 1

    wn_z = 1
    xi_z = 1

    wn_theta = 20 # higher than wn_x
    xi_theta = 1

    # x-z regulation:
    # x controller
    if xdd_ref is None:
        xdd_ref = 0
    v_theta = xdd_ref - 2*xi_x*wn_x*(X[3]-Xdes[3]) - wn_x**2*(X[0]-Xdes[0])
    th_ref = -1/grav_*v_theta

    # z controller
    if zdd_ref is None:
        zdd_ref = 0
    v_z = zdd_ref - 2*xi_z*wn_z*(X[4]-Xdes[4]) - wn_z**2*(X[1]-Xdes[1])
    dF = 1/m_*v_z

    # angle controller (slave to x)
    th = X[2]
    thd = X[5]
    thdd = 0 # for now, no feedforward terms!
    thd_ref = 0
    M = J_*(thdd - wn_theta**2*(th-th_ref) - 2*wn_theta*xi_theta*(thd-thd_ref))
    T = dF + m_*grav_

    return np.array([T, M]).astype(float)

# simulate
def simulate(ic, ctrl_fn, dt, tfinal, xtraj, use_csf=True):
    """
        simulates quadrotor output
        :param ic: initial condition, 6x1 vector
        :param ctrl_fn: control function, takes the state, desired state as arguments and returns inputs u1, u2
        :param dt
        :param tfinal
        :return: xlist, ulist for the quadrotor.
        """

    # clip the output
    Fmin = np.r_[-100, -100]
    Fmax = np.r_[100, 100]

    # create output lists
    xlist = np.zeros((math.floor(tfinal / dt), 6))
    ulist = np.zeros((math.floor(tfinal / dt), 2))
    times = np.arange(0, tfinal, dt)

    state = ic
    for i in range(len(times)):
        # calculate control
        e = state - xtraj[i,:]
        ministate = np.r_[e[0:2], e[3:5]]
        udes = -K@ministate
        xdd_des = udes[0,0]
        zdd_des = udes[0,1]

        # minimally invasive CSF
        if use_csf:
            utilde, success = calc_csf(state, np.array([xdd_des, zdd_des]), csf_fn)
            if not success:
                break
            xdd_des = utilde[0]
            zdd_des = utilde[1]

        # figure out the actual controls
        u = ctrl_fn(state, Xdes=xtraj[i,:], xdd_ref=xdd_des, zdd_ref=zdd_des)

        # don't bother with clipping the input for now.
        xlist[i, :] = state
        ulist[i, :] = u
        # print(u, state)
        state = (change_eqs(state, u.T)[:, 0] * dt + state).astype(float)

    return xlist, ulist, times


xlist, ulist, times = simulate(ic, ctrl_fn=hierarchical_ctrl, dt=dt, tfinal=tfinal, xtraj=xtraj, use_csf=USE_CSF)

fig = plt.figure()
xlim = np.max([abs(np.min(xlist[:,0])), np.max(xlist[:,0])])
ylim = np.max([abs(np.min(xlist[:,1])), np.max(xlist[:,1])])
lim = max([xlim, ylim])
lim = lim*1.1 # make 10% larger so the scaling is nice
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-lim, lim), ylim=(-lim, lim))
ax.grid()

# if flight == 'STRAIGHT_LINE':
#     # plot a circle centered at (1,.5)
phi = np.linspace(0,2*np.pi, 100)
circle_radius = .2
circle_x = circle_radius*np.cos(phi) + danger_x
circle_z = circle_radius*np.sin(phi) + danger_z
ax.plot(circle_x, circle_z)


line, = ax.plot([], [], 'o-', lw=10)
time_template = 'time = %.3fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):

    # calculate position of the engines
    center = np.r_[xlist[i,0], xlist[i,1]]
    th = xlist[i,2]
    R = np.array([[np.cos(th), -np.sin(th)],
                   [np.sin(th), np.cos(th)]])
    r.subs(subs_dict)
    lm_pos = center + (R @ np.r_[.127, 0])  # left motor position
    rm_pos = center - (R @ np.r_[.127, 0])  # right motor position

    thisx = [lm_pos[0], rm_pos[0]]
    thisy = [lm_pos[1], rm_pos[1]]

    # set line data
    line.set_data(thisx, thisy)
    time_text.set_text(time_template % (i*dt))
    return line, time_text

ani = animation.FuncAnimation(fig, animate, np.arange(len(times)), interval=1, blit=True, init_func=init)

if SAVE_ANIMATION:
    print("Writing animation...")
# ani.save('animation.gif', writer='imagemagick', fps=30)
    ani.save('no_csf_traj.mp4', writer="ffmpeg", fps=60)

#### plotting graphs ####

plt.figure()
plt.subplot(411)
for i, l in zip([0,1], ['x','z']):
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
plt.subplot(414, ylim=[-110,110])
for i, l in zip(range(2), ['F','M']):
    plt.plot(times, ulist[:,i], label=l)
legend = plt.legend(loc='upper right', shadow=True, fontsize='small')

show_traj = True
# show_traj = False
if show_traj:
    plt.figure()
    plt.plot(xlist[:,0], xlist[:,1], 'b.')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.plot(circle_x, circle_z)

plt.show()