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
    theta_0 = 0
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
        m_t = beta_1*m_t + (1-beta_1)*g_t
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)
        m_cap = m_t/(1-(beta_1**iter_num))
        v_cap = v_t/(1-(beta_2**iter_num))
        theta_0_prev = theta_0
        print("theta_0_prev", theta_0_prev)
        print("top", m_t)
        theta_0 = theta_0 - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)
        print("theta_0", theta_0)
        if(theta_0 == theta_0_prev):
            break
        res.append(linalg.norm(r)/linalg.norm(b))
    return res, n, x, "ADAM"
