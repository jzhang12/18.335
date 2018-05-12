import sys
sys.path.insert(0, '../algorithm')

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
		return 2*np.dot(np.transpose(A), b-A.dot(x))
	return toy_grad

def stochastic_toy_grad_i_fact(A, b, batch_size = 5):
	n, m = A.shape
	def stochastic_toy_grad_i(x):
		batch = np.random.choice(n, batch_size, replace = False)
		return 2*np.dot(np.transpose(A[batch]), (b[batch]-np.dot(A[batch], x)))
	return stochastic_toy_grad_i

def run_experiment(A, b, x, algo_string, data_set):
	obj = toy_obj_fact(A, b)
	if algo_string == "SGD":
		algo = sgd.sgd
		grad = stochastic_toy_grad_i_fact(A,b)
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
		grad = toy_grad_fact(A, b)
	else:
		print("INVALID ALGO")
		return
	res, n, x1, title = algo(obj, grad, x, A)
	title = data_set + " Dataset: "+ title + " Error vs Iteration"
	plt.plot(res)
	plt.title(title)
	plt.show()
