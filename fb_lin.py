import numpy as np
from numpy import pi, sin, cos
import math
import matplotlib.pyplot as plt
import sympy as sp
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import symbols, pprint, lambdify
import scipy.linalg
import numpy as np
import scipy.linalg


def lqr(A, B, Q, R):
    """Solve the continuous time lqr controller.

    dx/dt = A x + B u

    cost = integral x.T*Q*x + u.T*R*u
    """


    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R) * (B.T * X))

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals

def round_expr(expr, num_digits):
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(sp.Number)})

def lp(expr, prec=None, **kwargs):
    if prec is None:
        print(sp.latex(sp.simplify(expr), symbol_names=symbol_names, **kwargs))
def lpr(expr, prec, **kwargs):
    # latex print with rounding
    print(sp.latex(sp.simplify(round_expr(expr,prec)), symbol_names=symbol_names, **kwargs))

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

qd = sp.Matrix([xd,zd,thd])

D = sp.Matrix(np.diag([m, m, 1/3*m*r**2]))
u = sp.Matrix([F1, F2])
v = sp.Matrix([v1,v2])

B = sp.Matrix([[-sp.sin(th), -sp.sin(th)],
               [sp.cos(th), sp.cos(th)],
               [r, -r]])

H = np.diag([ox,oz,ot])@qd + sp.Matrix([ 0, -g, 0])
X = np.r_[x,z,th,xd,zd,thd]
subs_dict = {ox : 1e-6, oz : 1e-6, ot : 1e-6, r : .127, m : .1, g:9.8}

Dinv = sp.Matrix(np.diag([1/m, 1/m, 3/(m*r**2)]))
qdd = -Dinv@H + Dinv@B@u
S = np.array([[0,1,0],[0,0,1]])
ic = np.r_[0, 5, pi / 10, .1, 0, 0]

f = sp.Matrix([qd,Dinv@H]).subs(subs_dict)
g_ = sp.Matrix([[0,0],[0,0],[0,0],Dinv@B])
u = (S@Dinv@B).inv()@(S@Dinv@H + v)

f = f.subs(subs_dict)
g_ = g_.subs(subs_dict)
u = u.subs(subs_dict)

# control design
A = np.zeros()


xprime = sp.lambdify((X, v), f+g_@u)


# # euler's method
# dt = .001
# tfina l = 10
# times = np.arange(0,tfinal, dt)
#
# # simulate
# x = ic
# xdes = np.r_[0,5,0,0,0,0]
# xlist = np.zeros((math.floor(tfinal/dt), 6))
# ulist = np.zeros((math.floor(tfinal/dt), 2))
#
# show_traj = 1
# show_traj = 0
#
# for i in range(len(times)):
#
#     u =  np.clip(, Fmin, Fmax)
#
#     xlist[i,:] = x
#     ulist[i,:] = u
#     x = (f.subs(subs_dict)+g_.subs(subs_dict))*dt + x
#
# # make the plot
# plt.figure(1)
# plt.subplot(411)
# for i, l in zip(range(2), ['x','z']):
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
# #
# if show_traj:
#     plt.figure(2)
#     plt.plot(xlist[:,0], xlist[:,1], 'b.')
#     plt.xlabel('x')
#     plt.ylabel('z')
#
# plt.show()