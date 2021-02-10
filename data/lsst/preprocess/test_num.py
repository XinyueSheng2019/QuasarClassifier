import pandas as pd
import numpy as np 

data = pd.read_csv('plasticc_test_metadata.csv')

data = data[(data['ddf_bool']==1)&(data['true_target']==88)]
id_list = []
combine = data.groupby(data['object_id'])
for _id, group in combine:
	id_list.append(_id)

print(len(id_list))