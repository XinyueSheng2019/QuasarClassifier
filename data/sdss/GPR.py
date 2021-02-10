# Gaussian Process Regression visualization

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import json


obj_id = '8658170687885934766'

path = '../quasar_obs/'+obj_id+'.json'

mag = []
mag_err = []
time = []

# temp_time = [[52170, 52288], [52468, 52586], [52853, 52971], [53234, 53352], [53587, 53705], [53950, 54068], [54315, 54433]]
# x = np.atleast_2d(np.linspace(np.min(time),np.max(time),trange)).T
temp_time = [[54315, 54433]]
group_mjd = []
group_mag = []
group_err = []

with open(path,'r') as f:
	data=json.load(f)
	obs=data['obs']
	for ob in obs.values():
		mjd_mean=int(np.mean([band['mjd'] for band in ob.values() if band['mjd'] > 50000]))
		if mjd_mean in range(temp_time[0][0],temp_time[0][1],1):
			group_mjd.append(int(mjd_mean))
			group_mag.append(ob['g']['psfMag'])
			group_err.append(ob['g']['psfMagErr'])
		time.append(int(mjd_mean))
		mag.append(ob['g']['psfMag'])
		mag_err.append(ob['g']['psfMagErr'])



X = np.atleast_2d(group_mjd).T
y = np.array(group_mag).ravel()
noise = np.array(group_err)
y += noise

# kernel = C(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-1, 1e1))
kernel = C(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-1, 1e1))

gp = GaussianProcessRegressor(kernel=kernel, alpha = noise, n_restarts_optimizer=0)
gp.fit(X,y)

trange = np.max(time)-np.min(time)+1


x = []
for gap in temp_time:
	x += [t for t in range(gap[0],gap[1]+1,1)]

x = np.atleast_2d(np.array(x)).T


y_pred, sigma = gp.predict(x, return_std=True)

# print(len(list(y_pred)))




plt.figure(figsize = (5,5))
plt.errorbar(X.ravel(), y, noise, fmt='r.', markersize=5, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')

# fontsize        = 16
# labelsize       = fontsize
# tickwidth       = 2.0
# ticklength      = 2.0
# plt.tick_params(labelsize=labelsize, length=ticklength, width=tickwidth)
plt.xlabel('time(days)',fontsize= 16)
plt.ylabel('magnitude',fontsize= 16)

plt.legend(loc='lower right')

plt.show()




