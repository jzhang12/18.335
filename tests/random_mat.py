import numpy as np
import helper as h
import random
import matplotlib.pyplot as plt

n = 5

output = {}

def interpolate(time, data, interpolated_time):
	interpolated_data = []
	for i in interpolated_time:
		for t in xrange(len(time)):
			if i < time[t]:
				# point slope form
				m = float(data[t]-data[t-1])/(time[t]-time[t-1])
				interpolated_data.append(data[t-1] + m*(i-time[t-1]))
				break
			elif i == time[t]:
				interpolated_data.append(data[t])
				break
	return np.array(interpolated_data)

min_time = float('inf')
# Smoothing
for i in xrange(n):
	A = np.random.rand(20000,500)
	x_target = np.random.rand(500,1)
	b = np.dot(A,x_target)
	b +=  np.random.normal(0,1/np.linalg.norm(x_target),b.shape)
	x = np.random.rand(500,1)

	output[i] = h.run_experiment(A,b,x, "Random", plot = False)
	min_time = min(min_time, max(output[i]["SGD"][1]))
	min_time = min(min_time, max(output[i]["SCGD"][1]))
	min_time = min(min_time, max(output[i]["SNGD"][1]))
	min_time = min(min_time, max(output[i]["ADAM"][1]))

interpolated_time = np.arange(0, min_time, 0.1)
print interpolated_time

err, time, epoch = output[0]["SGD"]
sgd_err_epoch = np.array(err)/n
sgd_err_time = interpolate(time, err, interpolated_time)
sgd_err_time = sgd_err_time/n
for i in xrange(1, n):
	err, time, epoch = output[i]["SGD"]
	sgd_err_epoch += np.array(err)/n
	idata = interpolate(time, err, interpolated_time)
	sgd_err_time += idata/n

err, time, epoch = output[0]["SCGD"]
scgd_err_epoch = np.array(err)/n
scgd_err_time = interpolate(time, err, interpolated_time)
scgd_err_time = scgd_err_time/n
for i in xrange(1, n):
	err, time, epoch = output[i]["SCGD"]
	scgd_err_epoch += np.array(err)/n
	idata = interpolate(time, err, interpolated_time)
	scgd_err_time += idata/n

err, time, epoch = output[0]["SNGD"]
sngd_err_epoch = np.array(err)/n
sngd_err_time = interpolate(time, err, interpolated_time)
sngd_err_time = sngd_err_time/n
for i in xrange(1, n):
	err, time, epoch = output[i]["SNGD"]
	sngd_err_epoch += np.array(err)/n
	idata = interpolate(time, err, interpolated_time)
	sngd_err_time += idata/n

err, time, epoch = output[0]["ADAM"]
adam_err_epoch = np.array(err)/n
adam_err_time = interpolate(time, err, interpolated_time)
adam_err_time = adam_err_time/n
for i in xrange(1, n):
	err, time, epoch = output[i]["ADAM"]
	adam_err_epoch += np.array(err)/n
	idata = interpolate(time, err, interpolated_time)
	adam_err_time += idata/n

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

plt.plot(epoch, sgd_err_epoch, "b-", label = "SGD")
plt.plot(epoch, scgd_err_epoch, "r-", label = "SCGD")
plt.plot(epoch, sngd_err_epoch, "g-", label = "SNGD")
plt.plot(epoch, adam_err_epoch, "m-", label = "ADAM")

title_epoch = "Random Matrixes Dataset: Error vs Epoch"
plt.title(title_epoch)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

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

plt.plot(interpolated_time, sgd_err_time, "b-", label = "SGD")
plt.plot(interpolated_time, scgd_err_time, "r-", label = "SCGD")
plt.plot(interpolated_time, sngd_err_time, "g-", label = "SNGD")
plt.plot(interpolated_time, adam_err_time, "m-", label = "ADAM")

title_time = "Random Matrixes Dataset: Error vs Time"
plt.title(title_time)
plt.ylabel('Error')
plt.xlabel('Time')
plt.legend()
plt.show()

# sg_err_epochs = 