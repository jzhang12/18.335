import sys
sys.path.insert(0, '../algorithm')

import gd
import sgd
import cgd
import adam
import helper as h
import numpy as np
import matplotlib.pyplot as plt

# Approximate gradient
def finite_diff_fact(obj):
    def finite_diff(x, delta = 1e-3):
        x = x.reshape(x.shape[0],1)
        upper = np.repeat(x.T, len(x), 0) + delta*np.eye(len(x))
        lower = np.repeat(x.T, len(x), 0) - delta*np.eye(len(x))
        upper = np.apply_along_axis(lambda a: obj(a), 1, upper)
        lower = np.apply_along_axis(lambda a: obj(a), 1, lower)
        central = upper - lower
        central = (central.T / (2*delta))
        return central
    return finite_diff

def logistic_fact(A, b, l):
    n, m = A.shape
    A_temp = A
    A = np.ones((A.shape[0], A.shape[1]+1))
    A[:,1:] = A_temp
    def logistic_obj(w):
        return np.sum(np.log(1+np.exp(np.multiply(-1*b, np.dot(A, w).reshape(b.shape))))) + l*np.sum(w[1:]*w[1:])
    fd_grad = finite_diff_fact(logistic_obj)
    def logistic_grad(w):
        return fd_grad(w)
    def stochastic_logistic_grad(w, batch):
        def s_obj(w):
            return np.sum(np.log(1+np.exp(np.multiply(-1*b[batch], np.dot(A[batch], w).reshape(b[batch].shape))))) + l*np.sum(w[1:]*w[1:])
        stochastic_fd_grad = finite_diff_fact(s_obj)
        return stochastic_fd_grad(w)
    return logistic_obj, logistic_grad, stochastic_logistic_grad


# Define the predictLR(x) function, which uses trained parameters
def predictLR(X, w):
    return np.sign(1.0/(1+np.exp(-(np.dot(X,w[1:])+w[0])))-0.5).reshape(X.shape[0],1)

def scoreLR(predict, truth):
    results = (predict == truth)
    # indexes = np.where(predict!=truth)
    return float(np.sum(results))/truth.size

def score_fxn_fact(data, label):
    def score_fxn(params):
        predicted = predictLR(data, params)
        return scoreLR(predicted, label)
    return score_fxn

ds = np.loadtxt('data/mnist_digit_0.csv')
train = ds[:500,:]
test = ds[500:750,:]


ds = np.loadtxt('data/mnist_digit_1.csv')
train = np.vstack((train,ds[:500,:]))
test = np.vstack((test,ds[500:750,:]))

print "Finished Loading Data"

# Normalize
train = 2.0*train/255.0-1
test = 2.0*test/255.0-1

train_labels = np.vstack((np.ones((500, 1)), -1*np.ones((500, 1))))
test_labels = np.vstack((np.ones((250, 1)), -1*np.ones((250, 1))))

score_fxn = score_fxn_fact(train, train_labels)

obj, grad, sgrad = logistic_fact(train, train_labels, 0.0)
sgd_opt = h.stochastic_optimizer(train, train_labels, sgrad, batch_size = 256)
scgd_opt = h.stochastic_conj_gradient(train, train_labels, sgrad, batch_size = 512)
nesterov_opt = h.nesterov_optimizer(train, train_labels, sgrad, batch_size = 256)
adam_opt = h.adam_optimizer(train, train_labels, sgrad, batch_size = 256)

w_0 = np.zeros(train.shape[1]+1)
err_sgd, acc_sgd, times_sgd, epochs_sgd, title_sgd, w_sgd = sgd.sgd(obj, sgd_opt, w_0, score_fxn, num_epoch = 20)

w_0 = np.zeros(train.shape[1]+1)
err_sngd, acc_sngd, times_sngd, epochs_sngd, title_sngd, w_sngd = sgd.sngd(obj, nesterov_opt, w_0, score_fxn, num_epoch = 20)

w_0 = np.zeros(train.shape[1]+1)
err_adam, acc_adam, times_adam, epochs_adam, title_adam, w_adam = adam.adam(obj, adam_opt, w_0, score_fxn, num_epoch = 20)

w_0 = np.zeros(train.shape[1]+1)
err_scgd, acc_scgd, times_scgd, epochs_scgd, title_scgd, w_scgd = cgd.scgd(obj, scgd_opt, w_0, score_fxn, num_epoch = 20)

max_time = max(max(times_sgd), max(times_sngd), max(times_adam))

i = 0
while i < len(times_scgd):
    if times_scgd[i] > max_time:
        i += 1
        break
    i += 1
print i

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
title = "MNIST Dataset: " + " Training Loss vs Epoch"
plt.plot(epochs_sgd, err_sgd, "b-", label = title_sgd)
plt.plot(epochs_scgd, err_scgd, "r-", label = title_scgd)
plt.plot(epochs_sngd, err_sngd, "g-", label = title_sngd)
plt.plot(epochs_adam, err_adam, "m-", label = title_adam)
plt.legend()
plt.title(title)
plt.ylabel('Training Loss')
plt.xlabel('Epoch')
plt.show()

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
title = "MNIST Dataset: " + " Training Loss vs Time"
plt.plot(times_sgd, err_sgd, "b-", label = title_sgd)
plt.plot(times_scgd[:i], err_scgd[:i], "r-", label = title_scgd)
plt.plot(times_sngd, err_sngd, "g-", label = title_sngd)
plt.plot(times_adam, err_adam, "m-", label = title_adam)
plt.legend()
plt.title(title)
plt.ylabel('Training Loss')
plt.xlabel('Time')
plt.show()

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
title = "MNIST Dataset: " + " Training Accuracy vs Epoch"
plt.plot(epochs_sgd, acc_sgd, "b-", label = title_sgd)
plt.plot(epochs_scgd, acc_scgd, "r-", label = title_scgd)
plt.plot(epochs_sngd, acc_sngd, "g-", label = title_sngd)
plt.plot(epochs_adam, acc_adam, "m-", label = title_adam)
plt.legend()
plt.title(title)
plt.ylabel('Training Accuracy')
plt.xlabel('Epoch')
plt.show()

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)
title = "MNIST Dataset: " + " Training Accuracy vs Time"
plt.plot(times_sgd, acc_sgd, "b-", label = title_sgd)
plt.plot(times_scgd[:i], acc_scgd[:i], "r-", label = title_scgd)
plt.plot(times_sngd, acc_sngd, "g-", label = title_sngd)
plt.plot(times_adam, acc_adam, "m-", label = title_adam)
plt.legend()
plt.title(title)
plt.ylabel('Training Accuracy')
plt.xlabel('Time')
plt.show()

predict_test_sgd = predictLR(test, w_sgd)
print "SGD: ",
print scoreLR(predict_test_sgd, test_labels)
print

predict_test_sngd = predictLR(test, w_sngd)
print "SNGD: ",
print scoreLR(predict_test_sngd, test_labels)
print

predict_test_scgd = predictLR(test, w_scgd)
print "SCGD: ",
print scoreLR(predict_test_scgd, test_labels)
print

predict_test_adam = predictLR(test, w_adam)
print "ADAM: ",
print scoreLR(predict_test_adam, test_labels)
