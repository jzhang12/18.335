import numpy as np
import helper as h

A = np.random.rand(20,5)
x_target = np.random.rand(5,1)
b = np.dot(A,x_target)
x = np.random.rand(5,1)
# h.run_experiment(A,b,x, "SGD", "Random")
# h.run_experiment(A,b,x, "GD", "Random")
# h.run_experiment(A,b,x, "CGD", "Random")
h.run_experiment(A,b,x, "SCGD", "Random")
# h.run_experiment(A,b,x, "ADAM", "Random")