import math
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import copy

alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8


def adam(obj, grad, x, A, lr = 1e-4, tol = 1e-5, nmax = 1e6):
    theta_0 = np.zeros(len(x))
    theta_0_prev = np.ones(len(x))
    m_t = 0 
    v_t = 0 
    iter_num = 0

    r = obj(x)
    iter_num = 0
    res = [linalg.norm(r)]
    while iter_num < nmax:
        if res[iter_num] < tol:
            break
        iter_num+=1
        g_t = grad(x)
        for i, g in enumerate(g_t):
            m_t = beta_1*m_t + (1-beta_1)*g
            v_t = beta_2*v_t + (1-beta_2)*(g*g)
            m_cap = m_t/(1-(beta_1**iter_num))
            v_cap = v_t/(1-(beta_2**iter_num))
            theta_0_prev[i] = theta_0[i]
            theta_0[i] = theta_0[i] - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)
        res.append(linalg.norm(r))
        return res, iter_num, x, "ADAM"
