import numpy as np
import matplotlib.pyplot as plt
import copy

def line_search(grad, x, d, threshold = 1e-3):
    def g(a):
        return np.dot(np.transpose(d), grad(x+a*d))[0][0]
    lower_index = 0
    lower_bound = g(0)
    if abs(lower_bound) < threshold:
        return 0
    upper_index = 1
    upper_bound = g(1)
    if abs(upper_bound) < threshold:
        return 1
    init_search_range = 5
    while np.sign(lower_bound) == np.sign(upper_bound):
        for i in xrange(-init_search_range, init_search_range+1):
            upper_index = i
            upper_bound = g(i)
            if abs(upper_bound) < threshold:
                return i
            if np.sign(lower_bound) != np.sign(upper_bound):
                break
        init_search_range *= 2
    if upper_index < lower_index:
        upper_index, lower_index = lower_index, upper_index
        upper_bound, lower_bound = lower_bound, upper_bound
    while True:
        new_index = (upper_index+lower_index)/2.0
        new_bound = g(new_index)
        if abs(new_bound) < threshold:
            return new_index
        elif np.sign(lower_bound) == np.sign(new_bound):
            lower_bound = new_bound
            lower_index = new_index
        else:
            upper_bound = new_bound
            upper_index = new_index

def cgd(obj, grad, x, A, eps = 1e-7, nmax = 1e3):
    iter_num = 0
    res = [obj(x)]
    d = -grad(x)
    while iter_num < nmax:
        iter_num += 1
        alpha = line_search(grad, x, d)
        new_x = x + alpha*d
        old_grad = grad(x)
        new_grad = grad(new_x)
        grad_diff = new_grad-old_grad
        beta_pr = np.dot(np.transpose(new_grad), grad_diff)/np.dot(np.transpose(old_grad), old_grad)
        beta = max(beta_pr[0][0], 0)
        d = beta*d-new_grad
        x = new_x
        res.append(obj(x))
        if abs(res[iter_num]-res[iter_num-1]) < eps:
            break
    return res, iter_num, x, "CGD"

def scgd(obj, grad, x, A, batch_size = 2, eps = 1e-7, nmax = 1e2):
    n, m = A.shape
    iter_num = 0
    res = [obj(x)]
    d = -grad(x)
    while iter_num < nmax:
        iter_num += 1
        alpha = line_search(grad, x, d)
        new_x = x + alpha*d
        old_grad = grad(x)
        new_grad = grad(new_x)
        grad_diff = new_grad-old_grad
        beta_pr = np.dot(np.transpose(new_grad), grad_diff)/np.dot(np.transpose(old_grad), old_grad)
        beta = max(beta_pr[0][0], 0)
        d = beta*d-new_grad
        x = new_x
        res.append(obj(x))
        if abs(res[iter_num]-res[iter_num-1]) < eps:
            break
    return res, iter_num, x, "SGD"
