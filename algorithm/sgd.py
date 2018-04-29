import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import copy

def sgd(A, b, x, lr = 1e-2, eps = 1e-7, nmax = 1e3):
    r = b-A.dot(x)
    iter_num = 0
    res = [linalg.norm(r)]
    while iter_num < nmax:
        iter_num += 1
        x = x + 2*lr*np.dot(np.transpose(A),r)
        r = b-A.dot(x)
        res.append(linalg.norm(r))
        if abs(res[iter_num]-res[iter_num-1]) < eps:
            break
    return res, iter_num, x, "SGD"