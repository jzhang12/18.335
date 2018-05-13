import math
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import copy


alpha = 0.01
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8


def adam(obj, grad, init_x, lr = 1e-3, tol = 1e-5, nmax = 1e6):
    x = init_x
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    r = obj(x)
    res = [r]
    iter_num = 0
    while iter_num < nmax and (len(res)<2 or res[-2]-res[-1] > tol):
        dx = grad(x)
        iter_num +=1
        m = beta_1*m + (1-beta_1)*dx
        v = beta_2*v + (1-beta_2)*dx**2
        m_cap = m/(1-(beta_1**iter_num))
        v_cap = v/(1-(beta_2**iter_num))
        x = x - (alpha*m_cap)/(np.sqrt(v_cap) + epsilon)
        res.append(obj(x))
    
    return res, iter_num, x, "ADAM"
    