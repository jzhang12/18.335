import sys
sys.path.insert(0, '../algorithm')

import numpy as np
import matplotlib.pyplot as plt
import sgd

def run_experiment(A, b, x, algo):
    res, n, x1, title = algo(A, b, x)
    title = "Toy Dataset: "+title + " Error vs Iteration"
    plt.plot(res)
    plt.title(title)
    plt.show()

A = np.array([[1,1],[2,2],[3,3],[4,4]])
b = np.array([[1],[5],[5],[9]])
x = np.array([[0],[0]])
run_experiment(A,b,x, sgd.sgd)