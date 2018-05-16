import numpy as np
import matplotlib.pyplot as plt
import copy
import time

def sgd(obj, optimizer, x, score = None, lr = 1e-3, num_epoch = 20):
    iter_num = 0
    err = [obj(x)]
    if score is not None:
        acc = [score(x)]
    start = time.time()
    times = [0]
    epochs = [0]
    while iter_num < num_epoch:
        print "Epoch "+str(iter_num)+ ", Error: " + str(err[iter_num])
        iter_num += 1
        x = optimizer(x, lr)
        err.append(obj(x))
        if score is not None:
            acc.append(score(x))
        times.append(time.time() - start)
        epochs.append(iter_num)
    if score is not None:
        return err, acc, times, epochs, "SGD"
    return err, times, epochs, "SGD"


def sngd(obj, grad, x, score = None, lr = 1e-3, num_epoch = 20):
    iter_num = 0
    err = [obj(x)]
    if score is not None:
        acc = [score(x)]
    start = time.time()
    times = [0]
    epochs = [0]
    v = np.zeros(x.shape)
    v_prev = np.zeros(x.shape)
    prev_grads = np.zeros_like(x)
    while iter_num < num_epoch:
        print "Epoch "+str(iter_num)+ ", Error: " + str(err[iter_num])
        iter_num += 1
        x, prev_grads = grad(x, lr, prev_grads)
        err.append(obj(x))
        if score is not None:
            acc.append(score(x))
        times.append(time.time() - start)
        epochs.append(iter_num)
    if score is not None:
        return err, acc, times, epochs, "SNGD"
    return err, times, epochs, "SNGD"
