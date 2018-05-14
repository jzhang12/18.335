import numpy as np
import matplotlib.pyplot as plt
import copy

def sgd(obj, grad, x, A, lr = 1e-3, eps = 1e-3, nmax = 1e4):
    n, m = A.shape
    iter_num = 0
    res = [obj(x)]
    while iter_num < nmax:
        if iter_num%10 == 0:
            print "Iteration "+str(iter_num)+ " Accuracy: " + str(res[iter_num])
        x = x - lr*grad(x)
        res.append(obj(x))
        iter_num += 1
        if iter_num >= n:
            if abs(np.mean(res[iter_num-n+1:iter_num+1])-np.mean(res[iter_num-n:iter_num])) < eps:
                break
    return res, iter_num, x, "SGD"