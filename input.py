__author__ = "Xinyue Sheng"
__copyright__ = "Copyright (C) 2020 Xinyue Sheng"
__license__ = "Public Domain"
__version__ = "1.0"


from numpy.random import seed
# seed(1)
import tensorflow
# tensorflow.random.set_seed(1)
import numpy as np
import pandas as pd
import sys
import tensorflow.keras as ks 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C




def difference_data(id_list = list, data = list, features = list, group_bound = list):
	'''
	the difference between the neighboring magnitudes
	Inputs:
	- id_list: the id list of all objects
	- data: the data from the preprocessed file
	- features: the targeted feature list. Other features will be emitted
	- group_bound: group mjd intervals after setting each group to the same size
	Returns:
	- new_data: modified data

	'''

	new_data = data.copy()
	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		mjd_list = data['mjd'][ind_list].tolist()
		for bound in group_bound:
			bound_right = bound[1]
			bound_left = bound[0]
			group_mjd = [x for x in mjd_list if bound_left<=x<=bound_right]
			if len(group_mjd)>1:
				n = 0
				while n<len(group_mjd):
					if n == 0:
						idx = ind_list[0]+mjd_list.index(group_mjd[n])
						for feature in features:
							new_data.loc[idx,feature] = 0
					else:
						idx = ind_list[0]+mjd_list.index(group_mjd[n])
						for feature in features:
							new_data.loc[idx, feature] = data.loc[idx, feature] - data.loc[idx-1, feature]	

					n +=1

	return new_data 

	


def normalization(id_list = list, data = list, features = list):
	'''
	Rescaling (min-max normalization)
	Inputs:
	- id_list: the id list of all objects
	- data: the data from the preprocessed file
	- features: the targeted feature list. Other features will be emitted
	Returns:
	- new_data: modified data
	'''

	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		normalize_dict = {}
		for feature in features:
			value_list = data[feature][ind_list].tolist()
			_range = np.max(value_list) - np.min(value_list)
			_min = np.min(value_list)
			data.loc[ind_list, feature] = data.loc[ind_list, feature].map(lambda x: (x-_min)/_range)
			

	return data

def standardization(id_list = list, data = list, features = list):
	'''
	Standardization(Z-score normalization
	Inputs:
	- id_list: the id list of all objects
	- data: the data from the preprocessed file
	- features: the targeted feature list. Other features will be emitted
	Returns:
	- new_data: modified data
	'''

	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		normalize_dict = {}
		for feature in features:
			value_list = data[feature][ind_list].tolist()
			_mean = np.mean(value_list)
			_std = np.std(value_list)
			data.loc[ind_list, feature] = data.loc[ind_list, feature].map(lambda x: (x-_mean)/_std)

	return data


def GPR(id_list = list, data = list, group_bound = list, features = list, withErr = False):
	'''
	Guassian Process regression.
	Inputs:
	- id_list: the id list of all objects
	- data: the data from the preprocessed file
	- group_bound: group mjd intervals after setting each group to the same size
	- features: the targeted feature list. Other features will be emitted
	- withErr: True means add the magnitude errors into GP regression calculation
	Returns:
	- new_data: modified data
	'''
	print("start Gaussian process regression...")

	new_data = pd.DataFrame(columns = ['id','mjd']+features+['type'])

	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		mjd_list = data['mjd'][ind_list].tolist()
		fill_mjd_list = []
		obj_type = data['type'][ind_list[0]]
		insert_dataframe = pd.DataFrame(columns = ['id','mjd']+features+['type'])
		pred_dict = {}
		for f in features:
			pred_dict[f] = []
		for bound in group_bound:
			bound_left = bound[0]
			bound_right = bound[1]
			group_mjd = [x for x in mjd_list if bound_left<=x<=bound_right]
			group_mjd_idx = [mjd_list.index(x)+ind_list[0] for x in mjd_list if bound_left<=x<=bound_right]
			if len(group_mjd)>0:
				X = np.atleast_2d(group_mjd).T
				fill_mjd = np.atleast_2d(np.array([t for t in range(group_mjd[0],group_mjd[-1]+1,1)])).T
				group_fill_mjd = [t for t in range(group_mjd[0],group_mjd[-1]+1,1)]
				fill_mjd_list += group_fill_mjd

				for f in features:
					y = np.array(data[f][group_mjd_idx].tolist()).ravel()
					y_given = data[f][group_mjd_idx].tolist()
					noise = np.array(data[f+'_error'][group_mjd_idx].tolist())
					if withErr == True:
						y += noise
					kernel = C(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * RBF(length_scale=10, length_scale_bounds=(1e-1, 1e1))
					gp = GaussianProcessRegressor(kernel=kernel, alpha = noise)
					gp.fit(X,y)
					y_pred = list(gp.predict(fill_mjd))
					new_filled_values = []
					
					n = 0
					m = 0
					while(n<len(group_fill_mjd)):
						if group_fill_mjd[n] not in group_mjd:
							new_filled_values.append(y_pred[n])
						else:
							new_filled_values.append(y_given[m])
							m +=1
						n +=1
					pred_dict[f] += new_filled_values

		insert_dataframe['id'] = [i for t in fill_mjd_list]
		insert_dataframe['mjd'] = fill_mjd_list
		for f in features:
			insert_dataframe[f] = pred_dict[f]
		insert_dataframe['type'] = [obj_type for t in fill_mjd_list]
		new_data = new_data.append(insert_dataframe, ignore_index=True)

	print("GPR is Done.")

	return new_data



def remove_alone_mjd(id_list = list, data = list, check_delta = 300, min_size = 3):
	'''
	This function is used to remove the data points whose mjd is far away from other data points.
	These alone data pionts is not useful for group format and season format input data.
	Input:
	- id_list: the id list of all objects
	- data: the data from the preprocessed file
	- check_delta: if the difference between two neighboring data points' mjd is larger than this value, the earlier one will be removed.
	- min_size: the minimal size of the group before padding
	Returns:
	- data: the modified data
	'''

	record_row = []

	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		mjd_exit_list = data['mjd'][ind_list].tolist()
		warn_index = []
		n = 0
		while n+1<len(mjd_exit_list):
			delta = mjd_exit_list[n+1]-mjd_exit_list[n]

			if delta>check_delta:
				warn_index.append(ind_list[n])
			if n == 0 and delta>260:
				record_row.append(ind_list[0])
			n +=1
		if len(warn_index)!=0:
			t = 0
			while t+1<len(warn_index):
				if t == 0:
					if warn_index[0]-ind_list[0]<=min_size: record_row+=list(range(ind_list[0],warn_index[0]+1))

				if warn_index[t+1]-warn_index[t]<=min_size: 
					record_row += list(range(warn_index[t]+1, warn_index[t+1]+1))
				t +=1
			if len(warn_index)==1:
				if warn_index[0]-ind_list[0]<=min_size: record_row+=list(range(ind_list[0],warn_index[0]+1))


	new_data = data.drop(index=list(set(record_row)), axis=0).reset_index(drop=True)

	return new_data




def combine_narrow_mjd(id_list = list, data=list, check_delta = 0.7):
	'''
	This function is used for combine the data points whose mjds are closed. 
	For example, 2 data points in one mjd.
	Input:
	- id_list: the id list of all objects
	- data: the data from the preprocessed file
	- check_delta: if the difference between two neighboring data points' mjd is smaller than this value, the later one will be removed.
	Returns:
	- data: the modified data

	'''
	record_row = []
	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		mjd_exit_list = data['mjd'][ind_list].tolist()
		n = 0
		while n+1<len(mjd_exit_list):
			delta = mjd_exit_list[n+1]-mjd_exit_list[n]
			if delta<check_delta:
				record_row.append(ind_list[n])
			n +=1
	new_data = data.drop(index=list(set(record_row)), axis=0).reset_index(drop=True)
		
	return new_data


def make_int_mjd(data=list):
	'''
	change the mjd's format to integer
	'''
	
	data.loc[:,'mjd'] = data.loc[:,'mjd'].map(lambda x: int(x))
	return data


def group_observations(data = list, remove_check_delta = 233, combine_check_delta = 0.7, sequence_length = None, group_size = None, min_size = 3):
	'''
	This function is used to extract the main features of the data in order to group the observations
	It is called in format_group function.
	Inputs:
	- data: the data from the preprocessed file
	- remove_check_delta: used in remove_alone_mjd function. if the difference between two neighboring data points' mjd is larger than this value, the earlier one will be removed.
	- combine_check_delta: used in combine_narrow_mjd function. if the difference between two neighboring data points' mjd is smaller than this value, the later one will be removed.
	- sequence_length: the number of groups in one sequence
	- group_size: the fixed size of one group after padding
	- min_size: the minimal size of the group before padding
	Returns:
	- data: modified data
	- max_delta: the maximal gap between neighboring mjd data points
	- num_group: the fixed/designed number of group among all objects
	- max_group: the largest number of groups among all objects
	- min_group: the smallest number of groups among all objects
	- group_size: the fixed/designed size of group
	- modifid_group_bound: the modified group mjd intervals after setting each group to the same size
	'''

	combine = data.groupby(data['id']) 
	id_list = []
	for _id, group in combine:
	    id_list.append(_id)

	data =remove_alone_mjd(id_list, data, remove_check_delta, min_size)

	data = combine_narrow_mjd(id_list, data, combine_check_delta)

	delta_time = []
	num_group = []
	last_mjd = []
	size_group = []
	mjd_group = []

	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		mjd_exit_list = data['mjd'][ind_list].tolist()
		m = 0
		obj_delta_time = []
		while m<len(mjd_exit_list)-1:
			delta = mjd_exit_list[m+1]-mjd_exit_list[m]
			obj_delta_time.append(delta)
			m +=1
		delta_time.append(obj_delta_time)

		# the real time length of observations for each object
		last_mjd.append(int(mjd_exit_list[-1]))

		obs_time = mjd_exit_list[-1]-mjd_exit_list[0]

		# note: the num of groups is not equal to the observation year
		obj_num_group = len([x for x in obj_delta_time if x>=remove_check_delta])+1
		num_group.append(obj_num_group) 
		
		# calculate the mjd in each group for each object
		idx_group_gap = [obj_delta_time.index(x) for x in obj_delta_time if x>=remove_check_delta]

		# print(idx_group_gap)

		n = 0
		obj_group_size = []
		obj_group_mjd = []
		if len(idx_group_gap)>1:
			while n<len(idx_group_gap):
				if n == 0:
					obj_group_size.append(sum(obj_delta_time[:idx_group_gap[n]]))
					obj_group_mjd.append(mjd_exit_list[:idx_group_gap[n]+1])
				elif n == len(idx_group_gap)-1:
					obj_group_size.append(sum(obj_delta_time[idx_group_gap[n]+1:]))
					obj_group_mjd.append(mjd_exit_list[idx_group_gap[n]+1:])
				else:
					obj_group_size.append(sum(obj_delta_time[idx_group_gap[n]+1:idx_group_gap[n+1]]))
					obj_group_mjd.append(mjd_exit_list[idx_group_gap[n]+1:idx_group_gap[n+1]+1])

				n +=1
		elif len(idx_group_gap)==1:
			obj_group_size += [sum(obj_delta_time[:idx_group_gap[0]]), sum(obj_delta_time[idx_group_gap[0]+1:])]
			obj_group_mjd +=[mjd_exit_list[:idx_group_gap[0]+1], mjd_exit_list[idx_group_gap[0]+1:]]

		if len(idx_group_gap)==0:
			obj_group_size.append(sum(obj_delta_time))
			obj_group_mjd.append(mjd_exit_list)


		size_group.append(obj_group_size)
		mjd_group.append(obj_group_mjd)
	
	max_delta = int(np.max([np.max(x) for x in delta_time]))
	max_group = np.max(num_group)
	min_group = np.min(num_group)
	max_group_size = int(np.max([np.max(x) for x in size_group]))+1
	mean_group_size = int(np.mean([np.mean(x) for x in size_group]))+1

	print("maximal group number: ",max_group)
	print("minimal group number: ",min_group)
	print("maximal group size: ", max_group_size+1)
	print("mean group size: ", mean_group_size+1)

	# this mjd is regarded as the last mjd for every object
	base_mjd = np.max(last_mjd)
	print("the latest mjd is ", int(base_mjd)+1)


	#calculate the gap time between neigboring groups
	n = 0
	while n<len(id_list):
		group = num_group[n]
		obj_delta_time = delta_time[n]

		if group > 1 : 
			while len([x for x in obj_delta_time if x>=max_delta]) < group-1 and max_delta > 0:
				max_delta -= 1
		n +=1

	suit_delta = int(max_delta-1)
	print("the suitable gap time between each group: "+str(suit_delta))

	

	# calculate the boundary mjd of each group
	n = 1
	group_bound = []
	if sequence_length == None:
		sequence_length = max_group
	bound_left  = base_mjd - mean_group_size - suit_delta
	bound_right = base_mjd


	while n<=sequence_length:
		obj_bound_pre = []
		obj_bound_post = []
		
		for x in mjd_group:
			match = [t for t in x if bound_left<=t[0]<=bound_right] 
			if len(match)==1:
				obj_bound_pre.append(match[0][0])
				obj_bound_post.append(match[0][-1])

		if len(obj_bound_pre)>0 and len(obj_bound_post)>0:
			group_bound.append([np.min(obj_bound_pre),np.max(obj_bound_post)])
			gap = np.max(obj_bound_post)-np.min(obj_bound_pre)
			
			bound_left = bound_left - gap - suit_delta
			bound_right = bound_right - gap - suit_delta
		else:
			bound_left = bound_left - mean_group_size - suit_delta
			bound_right = bound_right - mean_group_size - suit_delta

		n +=1
	
	# print(group_bound)

	# modify the group size, and make it equal
	modified_group_bound = []
	if group_size == None:
		group_size = max_group_size
	for x in group_bound:
		new_bound = int(x[1]-group_size)
		modified_group_bound.append([new_bound,int(x[1])])


	
	modified_group_bound.reverse() 
	print(modified_group_bound)


	return data, max_delta, num_group, max_group, min_group, group_size, modified_group_bound


def format_season(data = list, train_id = list, test_id = list, features = list, sequence_length = None, remove_check_delta = 233, combine_check_delta = 0.7, group_size = None, min_size = 3, preprocess = 's', set_GPR = False):
	'''
	This function regard one season of observations for one object as one sequence.
	Therefore, one objects can offer several sequences.
	In this way, we could figure out the season trend of the variable object.
	The vector in each timestep is the combination of targeted bands.
	Inputs:
	- data: the data from the preprocessed file
	- train_id: the id of train objects
	- test_id: the id of test objects
	- features: the targeted feature list. Other features will be emitted.
	- sequence_length: the number of data points in one sequence
	- remove_check_delta: used in remove_alone_mjd function. if the difference between two neighboring data points' mjd is larger than this value, the earlier one will be removed.
	- combine_check_delta: used in combine_narrow_mjd function. if the difference between two neighboring data points' mjd is smaller than this value, the later one will be removed.
	- group_size: the fixed size of one group after padding
	- min_size: the minimal size of the group before padding
	- preprocess: choose the method of preprocessing: 
				  's': standardization
				  'n': normalization
				  'd': difference_data - set the difference between neighboring values as the preprocessed value
	- set_GPR: True means using Gaussian process regression to the magnitude data.
	Returns:
	- X_train/X_test: the train/test data with 3 dimensions: 
		objects, the number of observations in an objects, the vector in 1 observation
	- Y_train/Y_test: the label of train/test data. 
	'''


	data, max_delta, num_group, max_group, min_group, group_size, group_bound = group_observations(data = data, remove_check_delta = remove_check_delta, 
		combine_check_delta = combine_check_delta, sequence_length = sequence_length, group_size = group_size,  min_size = min_size)
	
	data = make_int_mjd(data)

	combine = data.groupby(data['id']) 
	id_list = []
	for _id, group in combine:
	    id_list.append(_id)

	if set_GPR == True:
		data = GPR(id_list, data, group_bound, features, withErr = False)

	if preprocess == 's':
		data = standardization(id_list, data, features)
	elif preprocess == 'n':
		data = normalization(id_list, data, features)
	elif preprocess == 'd':
		data = difference_data(id_list, data, features, group_bound)
	else:
		print('ERROR: wrong preprocess method input!')

	obj_data = []
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []


	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		mjd_list = data['mjd'][ind_list].tolist()
		n = 0
		
		for bound in group_bound:
			bound_left = bound[0]
			bound_right = bound[1]
			group_mjd = [x for x in mjd_list if bound_left<=x<=bound_right]
			if len(group_mjd)>=10:
				band_data = []
				while bound_left<=bound_right:
					if bound_left in group_mjd:
						idx = mjd_list.index(bound_left)+ind_list[0]
						band_data.append([data[x][idx] for x in features])
					else:
						band_data.append([0 for x in features])
					bound_left +=1
				if i in train_id:
					X_train.append(band_data)
					Y_train.append(data['type'][ind_list[0]])
				elif i in test_id:
					X_test.append(band_data)
					Y_test.append(data['type'][ind_list[0]])



	return X_train, Y_train, X_test, Y_test





def format_group(data = list, train_id = list, test_id = list, features = list, sequence_length = None, remove_check_delta = 233, combine_check_delta = 0.7, group_size = None, min_size = 3, preprocess = 's', set_GPR = False):
	'''
	This function groups the observations for one objects into groups.
	We assume that the data are discontinuous and that a large-scale observation is made once a year.
	Then, we could regard one year's observations as a group.
	In this way, we could figure out the year trend of the variable object.
	The length of the sequence is the number of groups.
	The vector in each timestep is the combination of a group's observation(one band's magnitudes differences)
	Inputs:
	- data: the data from the preprocessed file
	- train_id: the id of train objects
	- test_id: the id of test objects
	- features: the targeted feature list. Other features will be emitted.
	- sequence_length: the number of data points in one sequence
	- remove_check_delta: used in remove_alone_mjd function. if the difference between two neighboring data points' mjd is larger than this value, the earlier one will be removed.
	- combine_check_delta: used in combine_narrow_mjd function. if the difference between two neighboring data points' mjd is smaller than this value, the later one will be removed.
	- group_size: the fixed size of one group after padding
	- min_size: the minimal size of the group before padding
	- preprocess: choose the method of preprocessing: 
				  's': standardization
				  'n': normalization
				  'd': difference_data - set the difference between neighboring values as the preprocessed value
	- set_GPR: True means using Gaussian process regression to the magnitude data.
	
	Returns:
	- X_train/X_test: the train/test data with 3 dimensions: 
		objects, the number of observations in an objects, the vector in 1 observation
	- Y_train/Y_test: the label of train/test data. 

	'''

	data, max_delta, num_group, max_group, min_group, group_size, group_bound = group_observations(data = data, remove_check_delta = remove_check_delta, 
		combine_check_delta = combine_check_delta, sequence_length = sequence_length, group_size = group_size, min_size = min_size)

	data = make_int_mjd(data)

	combine = data.groupby(data['id']) 
	id_list = []
	for _id, group in combine:
	    id_list.append(_id)

	if set_GPR == True:
		data = GPR(id_list, data, group_bound, features, withErr = False)

	# normalization/ differences
	if preprocess == 's':
		data = standardization(id_list, data, features)
	elif preprocess == 'n':
		data = normalization(id_list, data, features)
	elif preprocess == 'd':
		data = difference_data(id_list, data, features, group_bound)
	else:
		print('ERROR: wrong preprocess method input!')


	obj_data = []
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []



	for i in id_list:
		ind_list = data[data.id == i].index.tolist()
		mjd_list = data['mjd'][ind_list].tolist()
		feature = features[0]
		value_dict = {}
		for f in features:
			value_dict[f] = data[f][ind_list].tolist()
		# value_list = data[feature][ind_list].tolist()

		n = 0
		obj_data = []

		for bound in group_bound:
			bound_right = bound[1]
			bound_left = bound[0]
			group_mjd = [x for x in mjd_list if bound_left<=x<=bound_right]

			if len(group_mjd)==0:
				sequence_data = [0 for x in range(0,(group_size+1)*len(features))]
				obj_data.append(sequence_data)
			else:
				sequence_data = []
				while bound_left<=bound_right:
					if bound_left in group_mjd:
						idx = mjd_list.index(bound_left)
						for f in features:
							sequence_data.append(value_dict[f][idx])
					else:
						for f in features:
							sequence_data.append(0)
					
					bound_left +=1

				obj_data.append(sequence_data)


		if i in train_id:
			X_train.append(obj_data)
			Y_train.append(data['type'][ind_list[0]])
		elif i in test_id:
			X_test.append(obj_data)
			Y_test.append(data['type'][ind_list[0]])

	

	return X_train, Y_train, X_test, Y_test






def format_simple(data = list, train_id = list, test_id = list, features = list, preprocess = 's'):
	'''
	This is the simple way of generating the input data.
	The length of the sequence is the biggest number of observations among all objects.
	The vector in each timestep is the combination of targeted features. eg.v1 = [u,g,r]
	Inputs:
	- data: the data from the preprocessed file
	- train_id: the id of train objects
	- test_id: the id of test objects
	- features: the targeted feature list. Other features will be emitted.
	- preprocess: choose the method of preprocessing: 
			  's': standardization
			  'n': normalization
			  'd': difference_data - set the difference between neighboring values as the preprocessed value
	Returns: 
	- X_train/X_test: the train/test data with 3 dimensions: 
		objects, the number of observations in an objects, the vector in 1 observation
	- Y_train/Y_test: the label of train/test data. 
	- test_list: the final object id order in the test set
	- test_type: the final object type order in the test set
	'''
	
	combine = data.groupby(data['id'])

	id_list = []
	for _id, group in combine:
	    id_list.append(_id)

	if preprocess == 's':
		data = standardization(id_list, data, features)
	elif preprocess == 'n':
		data = normalization(id_list, data, features)
	elif preprocess == 'd':
		data = difference_data(id_list, data, features, group_bound=[[np.min(data['mjd']),np.max(data['mjd'])]])
	else:
		print('ERROR: wrong preprocess method input!')

	obj_data = []
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	

	last_id = None
	last_type = None
	n = 0
	while n < len(data):
		obj_id = data['id'][n]
		if n == len(data)-1:
			obj_data.append([float(data[x][n]) for x in features])
		if obj_id != last_id or n == len(data)-1:
			if last_id in train_id:
				X_train.append(obj_data)
				if last_type != None:
					Y_train.append(last_type)
			elif last_id in test_id:
				X_test.append(obj_data)	
				if last_type != None:
					Y_test.append(last_type)
					

			obj_data = []
			last_id = obj_id
			last_type = data['type'][n]

		obj_data.append([float(data[x][n]) for x in features])
		n = n + 1

	return X_train, Y_train, X_test, Y_test


def load_data(path, test_fraction = 0.2, seed = None, features = list, set_format = 'group', preprocess = 's', group_size = None,
	min_group =3, sequence_length = None, remove_check_delta = 233, combine_check_delta = 0.7, set_GPR = False): 
	'''
	generate train/test set, divide the data into train and test sets.
	Inputs:
	- path: the processed file address
	- test fraction: float, the fraction of test set among all data
	- seed: int, when the seed is the same, the random result is the same
	- features: list, the features which will be fed into the neural network
	- set_format: choose an input format: simple, group, season
	- preprocess: choose the method of preprocessing: 
				  's': standardization
				  'n': normalization
				  'd': difference_data - set the difference between neighboring values as the preprocessed value
	- min_group: the minimal size of the group before padding
	- remove_check_delta: used in remove_alone_mjd function. if the difference between two neighboring data points' mjd is larger than this value, the earlier one will be removed.
	- combine_check_delta: used in combine_narrow_mjd function. if the difference between two neighboring data points' mjd is smaller than this value, the later one will be removed.
	- set_GPR: True means using Gaussian process regression to the magnitude data.
	Returns:
	- X_train/X_test: the train/test data with 3 dimensions: 
		objects, the number of observations in an objects, the vector in 1 observation
	- Y_train/Y_test: the label of train/test data. 
	- length_train/test: the number of sequences in the train/test set
	- time_sequence: the number of data points in one sequence
	- input_dim: the dimension of input data(train/test sets)
	- num_classes: the number of classes/labels

	'''

	print('Start constructing the input data...')
	
	last_id = None
	ids = []
	padding = False

	# Used to generate a specified random number
	np.random.seed(seed)

	# read the csv file
	data = pd.read_csv(path)  


	data = data[['id','mjd','type']+features]

	#remove NaN band values, replace NaN with 0
	data = data.dropna(subset = features, axis = 'rows', how='all').reset_index()
	data = data.fillna(0)


	# combine the data grouped by its id
	combine = data.groupby(data['id']) 

	# create the id list
	id_list = []
	for _id, group in combine:
		id_list.append(_id)

	# shuffle the index
	np.random.shuffle(id_list)

	# divide the id into test and train id lists
	test_len = int(len(id_list)*test_fraction)
	test_id = id_list[:test_len]
	train_id = id_list[test_len:]
	train_len = len(train_id)

	if set_format == 'group':
		X_train, Y_train, X_test, Y_test = format_group(data, train_id, test_id, features, min_size = min_group, sequence_length = sequence_length, remove_check_delta = remove_check_delta, combine_check_delta = combine_check_delta, group_size = group_size, preprocess = preprocess, set_GPR = set_GPR)
	elif set_format == 'season':
		X_train, Y_train, X_test, Y_test = format_season(data,train_id, test_id, features, min_size = min_group, sequence_length = sequence_length, remove_check_delta = remove_check_delta, combine_check_delta = combine_check_delta, group_size = group_size, preprocess = preprocess, set_GPR = set_GPR)
	elif set_format == 'simple':
		X_train, Y_train, X_test, Y_test = format_simple(data, train_id, test_id, features, preprocess = preprocess)
		padding = True
	else:
		print('ERROR: wrong format!')

	if padding == True:
		
		# obtain the max number of candidates among all objects, and pad the train and test sets to have the same number of candidates for each object
		maxlen = np.max([len(x) for x in X_train and X_test])
		X_train = ks.preprocessing.sequence.pad_sequences(X_train,maxlen=maxlen,padding='pre',value=0, dtype = 'float32')
		X_test = ks.preprocessing.sequence.pad_sequences(X_test,maxlen=maxlen,padding='pre',value=0, dtype = 'float32')

	else:
		# padding csv: convert to array
		X_train = np.array(X_train)
		X_test = np.array(X_test)

	num_classes = np.unique(Y_train).shape[0]

	Y_train = ks.utils.to_categorical(Y_train, num_classes = num_classes, dtype = 'float32')
	Y_test = ks.utils.to_categorical(Y_test, num_classes = num_classes, dtype = 'float32')

	length_train = X_train.shape[0]
	length_test = X_test.shape[0]

	time_sequence = X_train.shape[1]
	input_dim = X_train.shape[2]




	return (X_train, X_test, Y_train, Y_test),(length_train, length_test, time_sequence, input_dim, num_classes)


def cut_test_data(X_test, Y_test, cut_fraction, set_format, features):
	'''
	This function is used for testing the accuracy/AUC with the increasing number of observations in one group
	'''
	num_classes = np.unique(Y_test).shape[0]
	if cut_fraction == None or cut_fraction == 0 or set_format=='simple':
		if set_format != 'simple':
			X_test = np.array(X_test)
		else:
			maxlen = np.max([len(x) for x in X_test])
			X_test = ks.preprocessing.sequence.pad_sequences(X_test,maxlen=maxlen,padding='pre',value=np.nan)
		Y_test = ks.utils.to_categorical(Y_test, num_classes = num_classes, dtype = 'float32')

	elif float(cut_fraction)<1.0 and float(cut_fraction)>0:
		cut_fraction = float(cut_fraction)
		new_test = []
		if set_format == 'season':
			season_size = len(X_test[0])
			feature_num = len(features)
			save_num = int(season_size*(1-cut_fraction))
			for season in X_test:
				save = season[:save_num]
				remove = season[save_num:]
				zero_padding = [[0 for x in range(feature_num)] for y in range(len(remove))]
				new_test.append(save+zero_padding)

		elif set_format == 'group':
			group_size = int(len(X_test[0][0])/len(features))
			save_num = int(group_size*(1-cut_fraction))
			remove_num = group_size - save_num
			for obj in X_test:
				obj_sequence = []
				for group in obj:
					n = 0
					check = 0
					group_obs = []
					while n<len(features):
						group_obs +=group[check:check+save_num]+[0 for x in range(remove_num)]
						check += group_size
						n += 1
					obj_sequence.append(group_obs)
				new_test.append(obj_sequence)
		else:
			print('\nCut function doesn\'t support Simple input format.\n')

		X_test = new_test

		X_test = np.array(X_test)

		Y_test = ks.utils.to_categorical(Y_test, num_classes = num_classes, dtype = 'float32')

	return X_test, Y_test


def load_test_data(path, seed = None, features = list, set_format = 'group', preprocess = 's', min_group =3, group_size = None, sequence_length = None, remove_check_delta = 233, combine_check_delta = 0.7, set_GPR = False, padding = False):
	'''
	This function is used for the construction of test data. It will be used in the model prediction to test and compare the performance of different classifiers.
	'''
	print('Start constructing the test data...')
	np.random.seed(seed)
	data = pd.read_csv(path) 
	id_list = []
	combine = data.groupby(data['id']) 
	for _id, group in combine:
		id_list.append(_id)
	np.random.shuffle(id_list)

	if set_format == 'group':
		remove1, remove2, X_test, Y_test = format_group(data, [], id_list, features, min_size = min_group, sequence_length = sequence_length, remove_check_delta = remove_check_delta, combine_check_delta = combine_check_delta, group_size = group_size, preprocess = preprocess, set_GPR = set_GPR)
	elif set_format == 'season':
		remove1, remove2, X_test, Y_test = format_season(data, [], id_list, features, min_size = min_group, sequence_length = sequence_length, remove_check_delta = remove_check_delta, combine_check_delta = combine_check_delta, group_size = group_size, preprocess = preprocess, set_GPR = set_GPR)
	elif set_format == 'simple':
		remove1, remove2, X_test, Y_test = format_simple(data,  [], id_list, features, preprocess = preprocess)
	else:
		print('ERROR: wrong format!')


	return X_test, Y_test



	
if __name__ == '__main__':
	(X_train, X_test, Y_train, Y_test),(length_train, length_test, time_sequence, input_dim, num_classes) = load_data('../data/processed/unbalanced/test_set.csv', test_fraction = 0.2, seed = None, features = ['g','r'], set_format = 'simple', preprocess = 's', min_group =3, remove_check_delta = 233, combine_check_delta = 0.7, set_GPR = True)
	
	print(X_train)

	# test = sum(x[0] for x in Y_train)
	# test1 = sum(x[1] for x in Y_train)
	# print(test,' ', test1)



