# -*- coding: UTF-8 -*- 
from builtins import range
from builtins import object
import numpy as np
import linecache
import math
import os
import gzip
import copy
import numbers
import json as json
import matplotlib.pyplot as plt 
import matplotlib


def _simulate_agn(_agn_walk_start_date, expmjd, tau, time_dilation, sf_u, seed):
        """
    Simulate the u-band light curve for a single AGN

    Parameters
    ----------
    expmjd -- a number or numpy array of dates for the light curver

    tau -- the characteristic timescale of the AGN in days

    time_dilation -- (1+z) for the AGN

    sf_u -- the u-band structure function of the AGN

    seed -- the seed for the random number generator

    Returns
    -------
    a numpy array (or number) of delta_magnitude in the u-band at expmjd
    """

    # calculate the observed frame time

        if not isinstance(expmjd, numbers.Number):
            d_m_out = np.zeros(len(expmjd))
            duration_observer_frame = max(expmjd) - _agn_walk_start_date
        else:
            duration_observer_frame = expmjd - _agn_walk_start_date


        rng = np.random.RandomState(seed)
        dt = tau/100.
        duration_rest_frame = duration_observer_frame/time_dilation
        nbins = int(math.ceil(duration_rest_frame/dt))+1

        time_dexes = np.round((expmjd-_agn_walk_start_date)/(time_dilation*dt)).astype(int)
        time_dex_map = {}
        ct_dex = 0
        if not isinstance(time_dexes, numbers.Number):
            for i_t_dex, t_dex in enumerate(time_dexes):
                if t_dex in time_dex_map:
                    time_dex_map[t_dex].append(i_t_dex)
                else:
                    time_dex_map[t_dex] = [i_t_dex]
            time_dexes = set(time_dexes)
        else:
            time_dex_map[time_dexes] = [0]
            time_dexes = set([time_dexes])

        dx2 = 0.0
        x1 = 0.0
        x2 = 0.0

        dt_over_tau = dt/tau
        es = rng.normal(0., 1., nbins)*math.sqrt(dt_over_tau)
        for i_time in range(nbins):
            #The second term differs from Zeljko's equation by sqrt(2.) because he assumes stdev = sf_u/sqrt(2)
            dx1 = dx2
            dx2 = -dx1*dt_over_tau + sf_u*es[i_time] + dx1
            x1 = x2
            x2 += dt

            if i_time in time_dexes:
                if isinstance(expmjd, numbers.Number):
                    dm_val = ((expmjd-_agn_walk_start_date)*(dx1-dx2)/time_dilation+dx2*x1-dx1*x2)/(x1-x2)
                    d_m_out = dm_val
                else:
                    for i_time_out in time_dex_map[i_time]:
                        local_end = (expmjd[i_time_out]-_agn_walk_start_date)/time_dilation
                        dm_val = (local_end*(dx1-dx2)+dx2*x1-dx1*x2)/(x1-x2)
                        d_m_out[i_time_out] = dm_val

        return d_m_out

def obs_SF(mjd, flux):
    # convert mjd to integer
    mjd = mjd.astype(int).tolist()
#     flux = list(flux)
   
    #initialize the delta time
    delta = 1
    SF_list = []
    delta_list = []
    obs_len = max(mjd) - min(mjd)
    while delta < max(mjd)-min(mjd):
        n = min(mjd)
        mag_diff = []
        while n <= max(mjd)-delta:
            if n in mjd and n+delta in mjd: 
                mag_diff.append(flux[mjd.index(n+delta)] - flux[mjd.index(n)])
            n = n + 1
        if len(mag_diff) > 0:
            SF_list.append(np.var(mag_diff))
            delta_list.append(delta)  
        delta +=1
    
    plt.plot(delta_list, SF_list)
    plt.show()

def test():
    _agn_walk_start_date = 0
    expmjd = np.array(list(range(59580, 60680,3)))
    tau = float(input('input tau: '))
    redshift = float(input('input redshift:'))
    seed = int(input('input seed: '))
    sf = input('input SF, order by u,g,r,i,z,y: ').split(' ')
    sf = [float(x) for x in sf]
    time_dilation = redshift + 1
    sf_u = sf[0]
    sf_g = sf[1]
    sf_r = sf[2]
    sf_i = sf[3]
    sf_z = sf[4]
    sf_y = sf[5]

    # result is a list of flux differences
    result_u = _simulate_agn(_agn_walk_start_date, expmjd, tau, time_dilation, sf_u, seed)
    result_g = result_u*sf_g/sf_u
    result_r = result_u*sf_r/sf_u
    result_i = result_u*sf_i/sf_u
    result_z = result_u*sf_z/sf_u
    result_y = result_u*sf_y/sf_u


    # convert to the real flux, set the first flux first
    mag_record = np.zeros(len(expmjd))
    mag_record[0] = 0
    n = 1
    while n<len(result_u):
        mag_record[n] = mag_record[n-1] + result_u[n]
        n +=1


    # plt.plot(expmjd, mag_record)
    # plt.show()
    obs_SF(expmjd, mag_record)


test()




