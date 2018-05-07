import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import copy

def gd(obj, grad, x, A, lr = 1e-2, eps = 1e-7, nmax = 1e3):
    r = obj(x)
    iter_num = 0
    res = [linalg.norm(r)]
    while iter_num < nmax:
        iter_num += 1
        x = x + lr*grad(x)
        r = obj(x)
        res.append(linalg.norm(r))
        if abs(res[iter_num]-res[iter_num-1]) < eps:
            break
    return res, iter_num, x, "GD"