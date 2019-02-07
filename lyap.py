# Daniel Sun 06 Feb 2019 
# nonlineardynamics

import numpy as np
from numpy import pi, sin, cos
import math
import matplotlib.pyplot as plt
import sympy as sp
from sympy.interactive.printing import init_printing
init_printing(use_unicode=False, wrap_line=False)
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt
from sympy import symbols, pprint
import scipy.linalg



ox, oz, ot, m, r, F1, F2, g = symbols('ox, oz, ot, m, r, F1, F2, g')
kx, kz, kt, kxd, kzd, ktd= symbols('kx, kz, kt, kxd, kzd, ktd')
x, z, th, xd, zd, thd = symbols('x z th xd zd thd')

subs_dict = {ox : .01, oz : .01, ot : .0001, r : .127, m : .1, kt : .0001, kz : .0005}
symbol_names = {ox:'\sigma_x',
                oz:'\sigma_z',
                ot:'\sigma_\\theta',
                xd:'\dot{x}', zd:'\dot{z}', th:'\\theta', thd:'\dot{\\theta}'}

for k, ending in zip([kz, kt, kzd, ktd], ['z', 't','\dot{z}','\dot{\\theta}']):
    symbol_names[k] = "k_{" + ending + "}"

M = np.block([[np.zeros((3,3)), np.eye(3)],
              [np.zeros((3,3)), -np.diag([ox, oz, ot])]])



Fmin = np.r_[-100,-100]
Fmax = np.r_[100,100]

def B(X):
    theta = X[2]
    return 1/m*np.array([[0, 0],
                     [0, 0],
                     [0, 0],
                     [-sp.sin(theta), -sp.sin(theta)],
                     [sp.cos(theta), sp.cos(theta)],
                     [3/r, -3/r]]
             )

G = sp.Matrix([0,0,0,0,-g,0])

def xprime(M, B, G, x, u):
    """

    :param M: 6x6 mass matrix
    :param B: 6x2 matrix for input
    :param G: 6x1 matrix of gravity
    :param x: 6x1 current state of system
    :param u: 2x1 input effort
    :return: x', the rate of change of the system
    """

    return np.dot(M,x) + np.dot(B(x), u) + G

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



X = sp.Matrix([x, z, th, xd, zd, thd])
K = sp.Matrix([[0, kz, kt, 0, 0, 0],
              [0, kz, -kt, 0, 0, 0]])

def u(X):
    return 1/2*m*g/sp.cos(X[2])*sp.Matrix([1,1]) - K*X

xdot = M@X + B(X)@u(X) + G

# pprint(M)
# pprint(B(X))
# pprint(u(X))
# pprint(sp.simplify(M@X + B(X)@u(X)))
# pprint(sp.simplify(xdot))

def lp(expr, **kwargs):
    print(sp.latex(sp.simplify(expr), symbol_names=symbol_names,**kwargs))
# print(sp.latex(sp.simplify(xdot),  symbol_names=symbol_names, mode='equation'))

# lp(K)

Df = xdot.jacobian(X)
subs_dict = {x:0, z:0, th:0, xd:0, zd:0, thd:0,
             ox : .01, oz : .01, ot : .0001,
             r : .127, m : .1,
             kt : .0001, kz : .0005,
             g:9.8}
Df.evalf(subs=subs_dict)

A = np.array(Df.evalf(subs=subs_dict)).astype(np.float64)
A[4,2] = 0
lp(A)

lp(scipy.linalg.eigvals(A))
lp(scipy.linalg.solve_lyapunov(A, np.eye(6)))

