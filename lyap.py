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

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """


    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

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

# print(sp.latex(sp.simplify(xdot),  symbol_names=symbol_names, mode='equation'))
####################
# Defining symbols #
####################

ox, oz, ot, m, r, F1, F2, g = symbols('ox, oz, ot, m, r, F1, F2, g')
kx, kz, kt, kxd, kzd, ktd= symbols('kx, kz, kt, kxd, kzd, ktd')
x, z, th, xd, zd, thd = symbols('x z th xd zd thd')

symbol_names = {ox:'\sigma_x',
                oz:'\sigma_z',
                ot:'\sigma_\\theta',
                xd:'\dot{x}', zd:'\dot{z}', th:'\\theta', thd:'\dot{\\theta}'}

for k, ending in zip([kx,kz, kt, kxd, kzd, ktd,], ['x','z', 't','\dot{x}','\dot{z}','\dot{\\theta}']):
    symbol_names[k] = "k_{" + ending + "}"

M = np.block([[np.zeros((3,3)), np.eye(3)],
              [np.zeros((3,3)), -np.diag([ox, oz, ot])]])

U = sp.Matrix([F1,F2]).T

def B(X):
    theta = X[2]
    return 1/m*sp.Matrix([[0, 0],
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
K = sp.Matrix([[kx, kz, kt, kxd, kzd, ktd],
               [kx, kz, -kt, kxd, kzd, ktd]])

def u(X):
    return 1/2*m*g/sp.cos(X[2])*sp.Matrix([1,1]) - K*X

u_star = 1/2*m*g/sp.cos(X[2])*sp.Matrix([1,1])
F = sp.Matrix([F1,F2]).T
xdot = M@X + B(X)@(u_star + F.T) + G

#########################
# SYMBOLIC CALCULATIONS #
#########################

# pprint(M)
# pprint(B(X))
# pprint(u(X))
lp(B(X)@(u_star + K@X) + G)
lp(xdot)
XU = sp.Matrix([x, z, th, xd, zd, thd, F1,F2])

Df_wrtX = xdot.jacobian(X)
Df_wrtU = xdot.jacobian(U)
# lp(Df_wrtX)
# lp(Df_wrtU)
Fmin = np.r_[-100,-100]
Fmax = np.r_[100,100]

##########################
# Numerical Calculations #
##########################

subs_dict = {x:0, z:0, th:0, xd:0, zd:0, thd:0,
             ox : 1e-6, oz : 1e-6, ot : 1e-6,
             r : .127, m : .1,
             g:9.8, F1:0, F2:0}

A_ = np.array(Df_wrtX.subs(subs_dict)).astype(np.float64)
B_ = np.array(Df_wrtU.subs(subs_dict)).astype(np.float64)
Q = np.eye(6)
K,P,E = lqr(A_,B_,Q=Q,R=np.eye(2))
# lp(round_expr(sp.Matrix(K), 4))
w, vr = scipy.linalg.eig(A_, right=True)
Acl = A_-B_@K
P = scipy.linalg.solve_lyapunov(Acl, -Q) # negative because of how Q is defined
# lp(round_expr(sp.Matrix(P), 4))
P_eig, vr = scipy.linalg.eig(P, right=True)
Q_eig, vr = scipy.linalg.eig(Q, right=True)
# lpr(sp.Matrix(w), 5)


###################################
# Estimating domain of attraction #
###################################
max_eigP = max(P_eig)
min_eigQ = min(Q_eig)
print("max_ eig P = {}".format(max_eigP) + "min_eigQ = {}".format(min_eigQ))


# deriving G(x)
G_func = Df_wrtX - Df_wrtU@K

# factoring out X
for i in range(len(X)):
    G_func[:,i] = G_func[:,i]/X[i]

# substituting in dependence on F1, F2
subs_dict = {ox : 1e-6, oz : 1e-6, ot : 1e-6, r : .127, m : .1, g:9.8, F1:(K@X)[0,:], F2:(K@X)[1,:]}
G_func = G_func.subs(subs_dict)
G_func -= Acl # subtracting the closed loop function at the origin
total = 0
# G_func.simplify()
# computing matrix norm expression
for i in range(6):
    for j in range(6):
        print(type(G_func[i,j]))
        total += G_func[i,j]*G_func[i,j]
Gnorm = sp.sqrt(total)

##################################
# LaSalle's Invariance Principle #
##################################

vdot = 2*X.T@P@(M@X + B(X)@(u_star + -K@X) + G)
