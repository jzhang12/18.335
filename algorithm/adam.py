import math
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import copy
import time

# @profile
def adam(obj, grad, x, score = None, lr = 1e-3, num_epoch = 20):
    epoch_num = 0
    batch_num = 1
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    err = [obj(x)]
    if score is not None:
        acc = [score(x)]
    start = time.time()
    times = [0]
    epochs = [0]
    while epoch_num < num_epoch:
        print "Epoch "+str(epoch_num)+ ", Error: " + str(err[epoch_num])
        epoch_num += 1
        x, batch_num, m, v = grad(x, lr, batch_num, m, v)
        err.append(obj(x))
        if score is not None:
            acc.append(score(x))
        times.append(time.time() - start)
        epochs.append(epoch_num)
    if score is not None:
        return err, acc, times, epochs, "ADAM", x
    return err, times, epochs, "ADAM"