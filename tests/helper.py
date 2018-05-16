import sys
sys.path.insert(0, '../algorithm')

from numpy import *
from random import shuffle
from math import ceil
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import sgd
import gd
import cgd
import adam

def toy_obj_fact(A, b):
    def toy_obj(x):
        return linalg.norm(b - A.dot(x))
    return toy_obj

def toy_grad_fact(A, b):
    def toy_grad(x):
        return -2*np.dot(np.transpose(A), b-A.dot(x))
    return toy_grad

def stochastic_toy_grad_i_fact(A, b, batch_size = 256):
    n, m = A.shape
    ids = list(range(n))

    def stochastic_toy_grad_i(x):
        batch = np.random.choice(n, batch_size, replace = False)
        return -2*np.dot(np.transpose(A[batch]), (b[batch]-np.dot(A[batch], x)))/batch_size
    return stochastic_toy_grad_i

def gradient_fact(A,b):
    def batch_gradient(x, batch):
        return -2 * np.dot(np.transpose(A[batch]), (b[batch]-np.dot(A[batch], x)))
    return batch_gradient

def stochastic_optimizer(A, b, gradient, batch_size = 256):
    n, m = A.shape

    def stochastic_optimize_single(x, lr, start_grads = None):
        ids = list(range(n))
        shuffle(ids)

        for i in range(int(ceil(n / batch_size))):
            left = i * batch_size
            right = min((i + 1) * batch_size, n)
            batch = ids[left:right]
            grad = gradient(x,batch)/(right - left)
            x -= lr * grad
        return x
    return stochastic_optimize_single

def conj_gradient(A, b):
    n, m = A.shape

    def conj_gradient_single(x):
        return -2*np.dot(np.transpose(A), (b-np.dot(A, x)))
    return conj_gradient_single

def nesterov_optimizer(A, b, gradient, batch_size = 256, momentum = 0.9):
    n, m = A.shape

    def nesterov_optimize_single(x, lr, prev_grads):
        ids = list(range(n))
        shuffle(ids)

        for i in range(int(ceil(n / batch_size))):
            left = i * batch_size
            right = min((i + 1) * batch_size, n)
            batch = ids[left:right]
            delta = gradient(x,batch)/(right - left)
            grad = momentum * prev_grads - lr * delta
            x += -momentum * prev_grads + (1 + momentum) * grad
            prev_grads = grad

        return x, prev_grads
    return nesterov_optimize_single

def adam_optimizer(A, b, gradient, batch_size = 256, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8):
    n, m = A.shape

    def adam_optimize_single(x, lr, batch_num, m, v):
        ids = list(range(n))
        shuffle(ids)

        for i in range(int(ceil(n / batch_size))):
            left = i * batch_size
            right = min((i + 1) * batch_size, n)
            batch = ids[left:right]
            grad = gradient(x,batch)/(right - left)
            m = beta_1*m + (1-beta_1)*grad
            v = beta_2*v + (1-beta_2)*grad**2
            m_cap = m/(1-(beta_1**batch_num))
            v_cap = v/(1-(beta_2**batch_num))
            x = x - (lr*m_cap)/(np.sqrt(v_cap) + epsilon)
            batch_num += 1
        return x, batch_num, m, v
    return adam_optimize_single


def run_experiment(A, b, x, algo_string, data_set):
    obj = toy_obj_fact(A, b)
    if algo_string == "SGD":
        algo = sgd.sgd
        grad = stochastic_optimizer(A,b)
    elif algo_string == "GD":
        algo = gd.gd
        grad = toy_grad_fact(A, b)
    elif algo_string == "CGD":
        algo = cgd.cgd
        grad = toy_grad_fact(A, b)
    elif algo_string == "SCGD":
        algo = cgd.scgd
        grad = stochastic_toy_grad_i_fact(A, b)
    elif algo_string == "ADAM":
        algo = adam.adam
        grad = stochastic_toy_grad_i_fact(A, b)
    elif algo_string == "SNGD":
        algo = sgd.sngd
        grad = nesterov_optimizer(A, b)
    else:
        print("INVALID ALGO")
        return
    res, n, x1, title = algo(obj, grad, x, A)
    title = data_set + " Dataset: "+ title + " Error vs Iteration"
    plt.plot(res)
    plt.title(title)
    plt.show()

def run_experiments(A, b, x, algos_list, data_set):
    obj = toy_obj_fact(A, b)
    title_list = ""
    for algo_string in algos_list:
        if algo_string == "SGD":
            algo = sgd.sgd
            color = 'r-'
            gradient = gradient_fact(A,b)
            grad = stochastic_optimizer(A,b,gradient)
        elif algo_string == "SCGD":
            algo = cgd.scgd
            color = 'b-'
            gradient = gradient_fact(A,b)
            grad = stochastic_toy_grad_i_fact(A,b)
        elif algo_string == "ADAM":
            algo = adam.adam
            color = 'g-'
            gradient = gradient_fact(A,b)
            grad = adam_optimizer(A,b,gradient)
        elif algo_string == "SNGD":
            algo = sgd.sngd
            color = 'm-'
            gradient = gradient_fact(A,b)
            grad = nesterov_optimizer(A,b,gradient)
        else:
            print("INVALID ALGO")
            return
        err, times, epochs, title = algo(obj, grad, x)
        plt.plot(times, err, color, label = title)
        title_list += title + ", "
    title = data_set + " Dataset: "+ title_list+ "Error vs Iteration"
    plt.title(title)
    plt.ylabel('Error')
    plt.xlabel('Time')
    plt.legend()
    plt.show()