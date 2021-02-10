import numpy as np 
import pandas as pd 
import csv


def addrow(full_band_dict, object_id, mjd, wfile):
	new_row = [object_id, int(mjd)]
	for band in ['u','g','r','i','z','y']:
		if band not in full_band_dict:
			new_row += [None, None]
		else:
			new_row += full_band_dict[band]
	wfile.writerow(new_row)
	# print(new_row)

data = pd.read_csv('../code/rawdata/test_set_batch2.csv') #csv file need to be converted
meta = pd.read_csv('../code/rawdata/unblinded_test_set_metadata.csv') # metadata


meta = meta[(meta['ddf']<1)&(meta['target']==88)]
WDF_obj = meta.groupby('object_id')['object_id'].apply(list)

file = open('preprocessed_data/test_converted_AGN.csv','w')

wfile = csv.writer(file)
wfile.writerow(['id','mjd','u','u_err','g','g_err','r','r_err','i','i_err','z','z_err','y','y_err'])
band_dict = {0:'u',1:'g',2:'r',3:'i',4:'z',5:'y'}
last_id = None
last_mjd = None
full_band_dict = {}
n = 0
check = 0
while n < len(data):
	if data['object_id'][n] in WDF_obj:
		if data['object_id'][n]==last_id:
			delta_mjd = data['mjd'][n] - last_mjd
			if delta_mjd >=0.5:
				addrow(full_band_dict, last_id, last_mjd, wfile)
				full_band_dict = {}
				last_mjd = data['mjd'][n]
			if band_dict[data['passband'][n]] not in full_band_dict:
				full_band_dict[band_dict[data['passband'][n]]] = [data['flux'][n],data['flux_err'][n]]
		else:
			if last_id != None and last_mjd != None:
				addrow(full_band_dict, last_id, last_mjd, wfile)
			last_id = data['object_id'][n]
			last_mjd = data['mjd'][n]
			print(last_id)
			check +=1
			n -=1

	n +=1

print('check:', check)
print('test num:', len(WDF_obj))



