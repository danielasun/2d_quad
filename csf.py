import gurobipy as grb
from gurobipy import GRB
import numpy as np



m = grb.Model('qp')
m.Params.LogToConsole=0
u1 = m.addVar(lb=-GRB.INFINITY) # Thrust
u2 = m.addVar(lb=-GRB.INFINITY) # Force

def calc_csf(state, udes, calc_safety_coeffs):
    """
    calculate the control safety function according to Nonlinear Dynamics 28.23
    :param state: 6x1 state
    :param udes: 2x1 desired input
    :param calc_safety_coeffs: a function to calculate the control safety fn coefficients
    :return: u, 2x1 array for Thrust and Moment
    """
    Lfhx, Lghx, ah = calc_safety_coeffs(state)
    obj = (u1-udes[0])*(u1-udes[0]) + (u2-udes[1])*(u2-udes[1])
    m.setObjective(obj)

    m.addConstr(Lfhx + Lghx[0,0]*u1 + Lghx[0,1]*u2 +  ah >= 0, 'c0')

    try:
        m.optimize()
        print(m.x, udes)
    except:
        print("SAFETY CONSTRAINTS VIOLATED")
        return udes

    return np.array(m.x)



if __name__ == "__main__":
    from safety_fn_position import gen_safety_coeffs_fn

    csf_fn = calc_safety_coeffs = gen_safety_coeffs_fn(x0=1, z0=.5, alpha=1)
    state = np.array([.9, .4, .0, .1, -.2, .1])
    u1_des = 10
    u2_des = 6
    print(calc_csf(state, np.array([u1_des, u2_des]), csf_fn))