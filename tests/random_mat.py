import numpy as np
import helper as h

# A = np.random.rand(20,5)
# x_target = np.random.rand(5,1)
# b = np.dot(A,x_target)
# x = np.random.rand(5,1)
A = np.random.rand(200,50)
x_target = np.random.rand(50,1)
b = np.dot(A,x_target)
x = np.random.rand(50,1)
# h.run_experiment(A,b,x, "SGD", "Random")
# h.run_experiment(A,b,x, "SNGD", "Random")
# h.run_experiment(A,b,x, "GD", "Random")
# h.run_experiment(A,b,x, "CGD", "Random")
# h.run_experiment(A,b,x, "SCGD", "Random")
# h.run_experiment(A,b,x, "ADAM", "Random")
h.run_experiments(A,b,x, ["SGD", "SNGD", "ADAM"], "Random", iterations = 1e3)
# h.run_experiments(A,b,x, ["SNGD", "ADAM"], "Random", iterations = 1e2)
