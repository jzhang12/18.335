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

def stochastic_conj_gradient(A, b, gradient, batch_size = 256):
    n, m = A.shape

    def stochastic_conj_gradient_single(x):
        batch = np.random.choice(n, batch_size, replace = False)
        return gradient(x,batch)/batch_size
    return stochastic_conj_gradient_single

def nesterov_optimizer(A, b, gradient, batch_size = 256, momentum = 0.9):
    n, m = A.shape

    def nesterov_optimize_single(x, lr, prev_grads):
        ids = list(range(n))
        shuffle(ids)

        for i in range(int(ceil(n / batch_size))):
            left = i * batch_size
            right = min((i + 1) * batch_size, n)
            batch = ids[left:right]
            delta = gradient(x-momentum*prev_grads,batch)/(right - left)
            grad = momentum * prev_grads + lr * delta
            x += grad
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

def run_experiments(A, b, x, algos_list, data_set):
    obj = toy_obj_fact(A, b)
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
            grad = stochastic_conj_gradient(A,b)
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
        plt.plot(epochs, err, color, label = title)
    title = data_set + " Dataset: Error vs Epoch"
    plt.title(title)
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def run_experiment(A, b, x, data_set, plot = True):
    obj = toy_obj_fact(A, b)
    gradient = gradient_fact(A,b)
    sgd_grad = stochastic_optimizer(A,b,gradient)
    scgd_grad = stochastic_conj_gradient(A,b,gradient)
    adam_grad = adam_optimizer(A,b,gradient)
    sngd_grad = nesterov_optimizer(A,b,gradient)

    output = {}

    x_in = copy(x)
    sgd_err, sgd_times, sgd_epochs, sgd_title = sgd.sgd(obj, sgd_grad, x_in)
    output["SGD"] = (sgd_err, sgd_times, sgd_epochs)
    x_in = copy(x)
    scgd_err, scgd_times, scgd_epochs, scgd_title = cgd.scgd(obj, scgd_grad, x_in)
    output["SCGD"] = (scgd_err, scgd_times, scgd_epochs)
    x_in = copy(x)
    sngd_err, sngd_times, sngd_epochs, sngd_title = sgd.sngd(obj, sngd_grad, x_in)
    output["SNGD"] = (sngd_err, sngd_times, sngd_epochs)
    x_in = copy(x)
    adam_err, adam_times, adam_epochs, adam_title = adam.adam(obj, adam_grad, x_in)
    output["ADAM"] = (adam_err, adam_times, adam_epochs)

    if plot:
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 18
        
        title_epoch = data_set + " Dataset: Error vs Epoch"
        plt.title(title_epoch)
        plt.ylabel('Error')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

        max_time = max(max(sgd_times), max(sngd_times), max(adam_times))
        i = 0
        while i < len(scgd_times):
            if scgd_times[i] > max_time:
                i += 1
                break
            i += 1
        print i

        plt.plot(sgd_times, sgd_err, "b-", label = sgd_title)
        plt.plot(scgd_times[:i], scgd_err[:i], "r-", label = scgd_title)
        plt.plot(sngd_times, sngd_err, "g-", label = sngd_title)
        plt.plot(adam_times, adam_err, "m-", label = adam_title)

        title_time = data_set + " Dataset: Error vs Time"
        plt.title(title_time)
        plt.ylabel('Error')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

    return output