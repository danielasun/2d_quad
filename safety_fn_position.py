import sympy as sp
import numpy as np

from latex_util import *

# write down dynamics
x, z, th, xd, zd, thd = sp.symbols('x z th xd zd thd')
x0, z0 = sp.symbols('x0, z0')
gravity = sp.symbols('gravity')
ox, oz, ot, m, r, F1, F2, grav, v1, v2 = sp.symbols('ox, oz, ot, m, r, F1, F2, grav, v1, v2')
kx, kz, kt, kxd, kzd, ktd = sp.symbols('kx, kz, kt, kxd, kzd, ktd')

symbol_names = {ox: '\sigma_x',
                oz: '\sigma_z',
                ot: '\sigma_\\theta',
                xd: '\dot{x}', zd: '\dot{z}', th: '\\theta', thd: '\dot{\\theta}'}

for k, ending in zip([kx, kz, kt, kxd, kzd, ktd, ], ['x', 'z', 't', '\dot{x}', '\dot{z}', '\dot{\\theta}']):
    symbol_names[k] = "k_{" + ending + "}"

# Define dynamical system
subs_dict = {ox: 1e-6, oz: 1e-6, ot: 1e-6, r: .127, m: .1, grav: 9.8, x0:1, z0:.2}

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

# smaller, controllable system
A_ = np.array([[0,0,1,0],
               [0,0,0,1],
               [0,0,0,0],
               [0,0,0,0]])
B_ = np.array([[0,0],[0,0],[1,0],[0,1]])
miniX = sp.Matrix([x, z, xd, zd])
xdd_des, zdd_des = sp.symbols('xdd_des, zdd_des')
ftilde = A_@miniX
gtilde = B_

def H(arg):
    """
    Scaling function that satisfies

    Hmin <= H(arg) <= Hmax for all arg in R
    :param arg:
    :return: scaled value
    """
    return .005 + 1/(1+sp.exp(-arg))

def gen_safety_coeffs_fn(x0, z0, alpha):
    """
    Operates on the smaller 4x4 double integrator system with just x and z.

    h(x) = (x-x0)^2 + (z-z0)^2
    :param x0: x position of the center of the unsafe location
    :param z0: z position of the center of the unsafe location
    :param alpha:
    :return:
    """
    miniX = sp.Matrix([x, z, xd, zd])
    Lf1hx = sp.Matrix([2*(x-x0), 2*(z-z0), 0, 0]).T@ftilde
    hr = H(Lf1hx[0,0])*((x-x0)**2 + (z-z0)**2) # safety function that can only tell the thrusters to do more.
    Lfhx = sp.Matrix([hr]).jacobian(miniX)@ftilde
    Lghx = sp.Matrix([hr]).jacobian(miniX)@gtilde

    expr = Lfhx[0,0] + Lghx[0,0] + alpha*hr
    expr.subs({x0:1, z0:1})

    Lfhx = Lfhx.subs(subs_dict)[0,0]
    Lfhx = sp.lambdify(X, Lfhx)
    Lghx = Lghx.subs(subs_dict)
    Lghx = sp.lambdify(X, Lghx)
    hr = sp.lambdify(X, hr.subs(subs_dict))

    def calc_safety_coeffs(state):
        """
        calculate coefficients for CSF

        Right now, alpha=.1
        :param state: 6 vector of [x z th xd zd thd].T
        :return: Lfhx, Lghx (2 vec) .1*hr
        """

        print("hr = {}".format(hr(*state)))
        return Lfhx(*state), Lghx(*state), alpha*hr(*state)
    return calc_safety_coeffs

if __name__ == '__main__':
    calc_safety_coeffs = gen_safety_coeffs_fn(x0=1, z0=.5, alpha=1)
    state = np.array([.9, .4, .0, .1, -.2, .1])
    print(calc_safety_coeffs(state))