import numpy as np
import matplotlib.pyplot as plt
import copy

def gd(obj, grad, x, A, lr = 1e-3, eps = 1e-3, nmax = 1e2):
    iter_num = 0
    res = [obj(x)]
    while iter_num < nmax:
        print iter_num
        iter_num += 1
        x = x - lr*grad(x)
        res.append(obj(x))
        if abs(res[iter_num]-res[iter_num-1]) < eps:
            break
    return res, iter_num, x, "GD"