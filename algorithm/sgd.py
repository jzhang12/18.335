import numpy as np
import matplotlib.pyplot as plt
import copy

def sgd(obj, grad, x, A, lr = 1e-2, eps = 1e-7, nmax = 1e3):
    n, m = A.shape
    iter_num = 0
    res = [obj(x)]
    while iter_num < nmax:
        i = iter_num % n
        x = x + lr*grad(x, i)
        res.append(obj(x))
        iter_num += 1
        if iter_num >= n:
            if abs(np.mean(res[iter_num-n+1:iter_num+1])-np.mean(res[iter_num-n:iter_num])) < eps:
                break
    return res, iter_num, x, "SGD"
