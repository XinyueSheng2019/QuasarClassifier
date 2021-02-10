import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


data = pd.read_csv('processed_wtitle_c_gri_int.csv') 

mjd_list = data['mjd']

combine = mjd_list.groupby(mjd_list).count()

combine = list(combine)
print(combine)