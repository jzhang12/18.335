import numpy as np
import helper as h

A = np.array([[1,1],[2,2],[3,3],[4,4]])
b = np.array([[1],[5],[5],[9]])
x = np.array([[0],[0]])
h.run_experiment(A,b,x, "SGD", "Toy")
h.run_experiment(A,b,x, "GD", "Toy")
h.run_experiment(A,b,x, "CGD", "Toy")
h.run_experiment(A,b,x, "SCGD", "Toy")
h.run_experiment(A,b,x, "ADAM", "Toy")