import numpy as np
import helper as h
import random
import matplotlib.pyplot as plt

n = 3

output = {}

def interpolate(time, data, total_time, td):
	interpolated_data = []
	interpolated_time = []
	for i in np.arange(0, total_time+td, td):
		if i > time[len(time)-1]:
			return np.array(interpolated_data),np.array(interpolated_time)
		for t in xrange(len(time)):
			if i < time[t]:
				# point slope form
				m = float(data[t]-data[t-1])/(time[t]-time[t-1])
				interpolated_data.append(data[t-1] + m*(i-time[t-1]))
				interpolated_time.append(i)
				break
			elif i == time[t]:
				interpolated_data.append(data[t])
				interpolated_time.append(i)
				break
	return np.array(interpolated_data),np.array(interpolated_time)

min_time_sgd = float('inf')
min_time_scgd = float('inf')
min_time_sngd = float('inf')
min_time_adam = float('inf')
# Smoothing
for i in xrange(n):
	A = np.random.rand(20000,500)
	x_target = np.random.rand(500,1)
	b = np.dot(A,x_target)
	b +=  np.random.normal(0,1/np.linalg.norm(x_target),b.shape)
	x = np.random.rand(500,1)

	output[i] = h.run_experiment(A,b,x, "Random", plot = False)
	min_time_sgd = min(min_time_sgd, max(output[i]["SGD"][2]))
	min_time_scgd = min(min_time_scgd, max(output[i]["SCGD"][2]))
	min_time_sngd = min(min_time_sngd, max(output[i]["SNGD"][2]))
	min_time_adam = min(min_time_adam, max(output[i]["ADAM"][2]))

err, time, epoch = output[0]["SGD"]
sgd_err_epoch = np.array(err)/n
sgd_err_time, sgd_interpolated_time = interpolate(time, err, min_time_sgd, 0.1)
sgd_err_time = sgd_err_time/n
for i in xrange(1, n):
	err, time, epoch = output[i]["SGD"]
	sgd_err_epoch += np.array(err)/n
	idata, itime = interpolate(time, err, min_time_sgd, 0.1)
	sgd_err_time += idata/n

err, time, epoch = output[0]["SCGD"]
scgd_err_epoch = np.array(err)/n
scgd_err_time, scgd_interpolated_time = interpolate(time, err, min_time_scgd, 0.1)
scgd_err_time = scgd_err_time/n
for i in xrange(1, n):
	err, time, epoch = output[i]["SCGD"]
	scgd_err_epoch += np.array(err)/n
	idata, itime = interpolate(time, err, min_time_scgd, 0.1)
	scgd_err_time += idata/n

err, time, epoch = output[0]["SNGD"]
sngd_err_epoch = np.array(err)/n
sngd_err_time, sngd_interpolated_time = interpolate(time, err, min_time_sngd, 0.1)
sngd_err_time = sngd_err_time/n
for i in xrange(1, n):
	err, time, epoch = output[i]["SNGD"]
	sngd_err_epoch += np.array(err)/n
	idata, itime = interpolate(time, err, min_time_sngd, 0.1)
	sngd_err_time += idata/n

err, time, epoch = output[0]["ADAM"]
adam_err_epoch = np.array(err)/n
adam_err_time, adam_interpolated_time = interpolate(time, err, min_time_adam, 0.1)
adam_err_time = adam_err_time/n
for i in xrange(1, n):
	err, time, epoch = output[i]["ADAM"]
	adam_err_epoch += np.array(err)/n
	idata, itime = interpolate(time, err, min_time_adam, 0.1)
	adam_err_time += idata/n


plt.plot(epoch, sgd_err_epoch, "b-", label = sgd_title)
plt.plot(epoch, scgd_err_epoch, "r-", label = scgd_title)
plt.plot(epoch, sngd_err_epoch, "g-", label = sngd_title)
plt.plot(epoch, adam_err_epoch, "m-", label = adam_title)

title_epoch = data_set + " Dataset: Error vs Epoch"
plt.title(title_epoch)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.legend()
plt.show()

max_time = max(max(sgd_interpolated_time), max(sngd_interpolated_time), max(adam_interpolated_time))
i = 0
while i < len(scgd_times):
    if scgd_interpolated_time[i] > max_time:
        i += 1
        break
    i += 1
print i

plt.plot(sgd_interpolated_time, sgd_err_time, "b-", label = sgd_title)
plt.plot(scgd_interpolated_time[:i], scgd_err_time[:i], "r-", label = scgd_title)
plt.plot(sngd_interpolated_time, sngd_err_time, "g-", label = sngd_title)
plt.plot(adam_interpolated_time, adam_err_time, "m-", label = adam_title)

title_time = data_set + " Dataset: Error vs Time"
plt.title(title_time)
plt.ylabel('Error')
plt.xlabel('Time')
plt.legend()
plt.show()


plt.show()

# sg_err_epochs = 