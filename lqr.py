import numpy as np
from numpy import pi, sin, cos
import math
import matplotlib.pyplot as plt
import sympy as sym

M = np.block([[np.zeros((3,3)), np.eye(3)],[np.zeros((3,3)), np.zeros((3,3))]])

r = .127 # 5 inches => meters
m = .1 # kg
Fmin = np.r_[-100,-100]
Fmax = np.r_[100,100]

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
