import numpy as np
import matplotlib.pyplot as plt
import copy

def gd(obj, grad, x, A, lr = 1e-4, eps = 1e-2, nmax = 3e2):
    iter_num = 0
    res = [obj(x)]
    while iter_num < nmax:
        if iter_num%10 == 0:
            print "Iteration "+str(iter_num)+ " Accuracy: " + str(res[iter_num])
        iter_num += 1
        x = x - lr*grad(x)
        res.append(obj(x))
        if abs(res[iter_num]-res[iter_num-1]) < eps:
            break
    return res, iter_num, x, "GD"