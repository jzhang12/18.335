import sys
sys.path.insert(0, '../algorithm')

import gd
import sgd
import cgd
import adam
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

def logistic_fact(A, b, l, batch_size = 10):
    n, m = A.shape
    A_temp = A
    A = np.ones((A.shape[0], A.shape[1]+1))
    A[:,1:] = A_temp
    def logistic_obj(w):
        return np.sum(np.log(1+np.exp(np.multiply(-1*b, np.dot(A, w).reshape(b.shape))))) + l*np.sum(w[1:]*w[1:])
    fd_grad = finite_diff_fact(logistic_obj)
    def logistic_grad(w):
        return fd_grad(w)
    def stochastic_logistic_grad(w):
        batch = np.random.choice(n, batch_size, replace = False)
        def s_obj(w):
            return np.sum(np.log(1+np.exp(np.multiply(-1*b[batch], np.dot(A[batch], w).reshape(b[batch].shape))))) + l*np.sum(w[1:]*w[1:])
        stochastic_fd_grad = finite_diff_fact(s_obj)
        return stochastic_fd_grad(w)
    return logistic_obj, logistic_grad, stochastic_logistic_grad

pos = ['0', '2', '4', '6', '8']
neg = ['1', '3', '5', '7', '9']
# pos = ['0']
# neg = ['1']

ds = np.loadtxt('data/mnist_digit_'+pos[0]+'.csv')
train = ds[:200,:]
# dev = ds[200:350,:]
test = ds[350:500,:]

# load data from csv files
for num in pos[1:]:
    ds = np.loadtxt('data/mnist_digit_'+num+'.csv')
    train = np.vstack((train,ds[:200,:]))
    # dev = np.vstack((dev,ds[200:350,:]))
    test = np.vstack((test,ds[350:500,:]))

for num in neg:
    ds = np.loadtxt('data/mnist_digit_'+num+'.csv')
    train = np.vstack((train,ds[:200,:]))
    # dev = np.vstack((dev,ds[200:350,:]))
    test = np.vstack((test,ds[350:500,:]))

print "Finished Loading Data"

train = 2.0*train/255.0-1
# dev = 2.0*dev/255.0-1
test = 2.0*test/255.0-1

train_labels = np.vstack((np.ones((200*len(pos), 1)), -1*np.ones((200*len(neg), 1))))
# dev_labels = np.vstack((np.ones((150*len(pos), 1)), -1*np.ones((150*len(neg), 1))))
test_labels = np.vstack((np.ones((150*len(pos), 1)), -1*np.ones((150*len(neg), 1))))

obj, grad, sgrad = logistic_fact(train, train_labels, 0.0)
w_0 = np.zeros(train.shape[1]+1)

# res, n, x1, title = gd.gd(obj, grad, w_0, train)
# res, n, x1, title = cgd.cgd(obj, grad, w_0, train)
res, n, x1, title = sgd.sgd(obj, sgrad, w_0, train, nmax = 1e2)
# res, n, x1, title = cgd.scgd(obj, sgrad, w_0, train)
res1, n, x1, title = sgd.sngd(obj, sgrad, w_0, train, nmax = 1e2)
res2, n, x1, title = adam.adam(obj, sgrad, w_0, train, nmax = 1e2)

title = "MNIST Dataset: "+ title + " Error vs Iteration"
plt.plot(res)
plt.plot(res1)
plt.plot(res2)
plt.title(title)
plt.show()

# Define the predictLR(x) function, which uses trained parameters
def predictLR(X, w):
    return np.sign(1.0/(1+np.exp(-(np.dot(X,w[1:])+w[0])))-0.5).reshape(X.shape[0],1)

def scoreLR(predict, truth):
    results = (predict == truth)
    indexes = np.where(predict!=truth)
    return float(np.sum(results))/truth.size, indexes

predict_train = predictLR(train, x1)
print scoreLR(predict_train, train_labels)[0]
predict_test = predictLR(test, x1)
print scoreLR(predict_test, test_labels)[0]