import math
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import copy




# def adam(obj, grad, x, A, lr = 1e-4, tol = 1e-5, nmax = 1e6):
#     theta_0 = np.zeros(len(x))
#     # theta_0_prev = np.zeros(len(x))
#     m_t = np.zeros(len(x))
#     v_t = np.zeros(len(x))
#     m_cap = np.zeros(len(x))
#     v_cap = np.zeros(len(x))
#     iter_num = 0

#     r = obj(x)
#     iter_num = 0
#     res = [linalg.norm(r)]
#     while iter_num < nmax:
#         if res[iter_num] < tol:
#             break
#         iter_num+=1
#         g_t = grad(x)
#         for i, g in enumerate(g_t):
#             m_t[i] = beta_1*m_t[i] + (1-beta_1)*g
#             v_t[i] = beta_2*v_t[i] + (1-beta_2)*(g*g)
#             m_cap = m_t[i]/(1-(beta_1**iter_num))
#             v_cap = v_t[i]/(1-(beta_2**iter_num))
#             theta_0[i] = theta_0_prev[i] - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)
#             theta_0_prev[i] = theta_0[i]
#         res.append(linalg.norm(r))
#         return res, iter_num, x, "ADAM"

alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8


def adam(obj, grad, init_x, lr = 1e-4, tol = 1e-5, nmax = 1e6):
    x = init_x
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    r = obj(x)
    res = [r]
    iter_num = 0
    while iter_num < nmax and res[-1] > tol:
        dx = grad(x)
        iter_num +=1
        m = beta_1*m + (1-beta_1)*dx
        v = beta_2*v + (1-beta_2)*dx**2
        m_cap = m/(1-(beta_1**iter_num))
        v_cap = v/(1-(beta_2**iter_num))
        x = x - (alpha*m_cap)/(np.sqrt(v_cap) + epsilon)
        res.append(obj(x))
    
    return res, iter_num, x, "ADAM"
    
    
    