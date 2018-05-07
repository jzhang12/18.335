import math
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import copy

alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

theta_0 = 0
m_t = 0 
v_t = 0 
t = 0

def func(A, x):
    return A.dot(x)

def grad_func(A, x):#calculates the gradient
    return A

def adam(A, x, b, lr = 1e-4, tol = 1e-5, nmax = 1e6):
    r = b-A.dot(x)
    iter_num = 0
    res = [linalg.norm(r)/linalg.norm(b)]
    while iter_num < nmax:
        if res[iter_num] < tol:
            break
        iter_num+=1
        g_t = grad_func(A, theta_0)
        m_t = beta_1*m_t + (1-beta_1)*g_t
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t)
        m_cap = m_t/(1-(beta_1**t))
        v_cap = v_t/(1-(beta_2**t))
        theta_0_prev = theta_0
        theta_0 = theta_0 - (alpha*m_cap)/(math.sqrt(v_cap)+epsilon)
        print(theta_0)
        if(theta_0 == theta_0_prev):
            break
        res.append(linalg.norm(r)/linalg.norm(b))
    return res, n, x, "ADAM"