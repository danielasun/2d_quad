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


def round_expr(expr, num_digits):
    return expr.xreplace({n: round(n, num_digits) for n in expr.atoms(sp.Number)})


def lp(expr, prec=None, **kwargs):
    if prec is None:
        print(sp.latex(sp.simplify(expr), symbol_names=symbol_names, **kwargs))


def lpr(expr, prec, **kwargs):
    # latex print with rounding
    print(sp.latex(sp.simplify(round_expr(expr, prec)), symbol_names=symbol_names, **kwargs))


# write down dynamics
ox, oz, ot, m, r, F1, F2, g, v1, v2 = symbols('ox, oz, ot, m, r, F1, F2, g, v1, v2')
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
subs_dict = {ox: 1e-6, oz: 1e-6, ot: 1e-6, r: .127, m: .1, g: 9.8}

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
H = sp.Matrix([0, m * g, 0])

Dinv = sp.Matrix(np.diag([1 / m, 1 / m, 1 / J]))
qdd = -Dinv @ H + Dinv @ B @ u

f = sp.Matrix([qd, -Dinv @ H]).subs(subs_dict)
g = sp.Matrix([[0, 0], [0, 0], [0, 0], Dinv @ B])

# create change equations
change_eqs = sp.lambdify([X, u], (f + g @ u).subs(subs_dict))


fm_to_u1u2 = sp.Matrix([[1/2, 1/(2*r)],
                        [1/2, -1/(2*r)]]).subs(subs_dict)

def ctrl_fn(X, Xdes):
    """
    State dependent control
    :param X:
    :return: 2x1 control output vector
    """
    return np.r_[5, .1]


# simulate
def simulate(ic, ctrl_fn, dt, tfinal):
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
        u = ctrl_fn(X, Xdes=np.r_[0, 0, 0, 0, 0, 0])

        # don't bother with clipping the input for now.

        xlist[i, :] = state
        ulist[i, :] = u
        state = change_eqs(state, u.T)[:, 0] * dt + state
    return xlist, ulist, times

# make the plot
ic = np.r_[1, 1, -.5, 1, 1, .1]
dt = .01
xlist, ulist, times = simulate(ic, ctrl_fn=ctrl_fn, dt=dt, tfinal=1)


fig = plt.figure()
xlim = np.max([abs(np.min(xlist[:,0])), np.max(xlist[:,0])])
ylim = np.max([abs(np.min(xlist[:,1])), np.max(xlist[:,1])])
lim = max([xlim, ylim])
lim = lim*1.1 # make 10% larger so the scaling is nice
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-lim, lim), ylim=(-lim, lim))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
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

ani = animation.FuncAnimation(fig, animate, np.arange(len(times)),
                              interval=25, blit=False, init_func=init)

# ani.save('quadrotor.mp4', fps=15)
plt.show()

#### plotting graphs ####

# plt.figure(1)
# plt.subplot(411)
# for i, l in zip([0,1], ['x','z']):
#     plt.plot(times, xlist[:,i], label=l)
# legend = plt.legend(loc='lower right', shadow=True, fontsize='small')
# plt.xlabel('time')
# plt.title('position vs. time')
# plt.subplot(412)
# plt.plot(times, xlist[:,2], label='theta')
# plt.xlabel('time')
# plt.title('theta vs. time')
# plt.subplot(413)
# for i, l in zip(range(3,6), ['xdot','zdot','thetadot']):
#     plt.plot(times, xlist[:,i], label=l)
# plt.xlabel('time')
# legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
# plt.subplot(414)
# for i, l in zip(range(2), ['F1','F2']):
#     plt.plot(times, ulist[:,i], label=l)
# legend = plt.legend(loc='upper right', shadow=True, fontsize='small')
#
# show_traj = True
# if show_traj:
#     plt.figure(2)
#     plt.plot(xlist[:,0], xlist[:,1], 'b.')
#     plt.xlabel('x')
#     plt.ylabel('z')

# plt.show()




###### not using right now
# # augmented system: X_d = f_(x,u) + g_@ud
# show stabilization to the origin

# show general trajectory tracking for the controller.

# def kelly_stabilizer(z, p):
#     """
#     Nonlinear control law that will stabiize the quadrotor.
#     :param z: 6 vector for the state
#     :param p:
#     :return:
#     """
#
#     wfast = 1
#     wslow_x = .1
#     wslow_z = .1
#     xi = .1
#
#     ddx =

# formulate barrier function
# formulate
