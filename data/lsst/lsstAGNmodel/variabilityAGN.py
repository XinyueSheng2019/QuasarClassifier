
from builtins import range
from builtins import object
import numpy as np
import linecache
import math
import os
import gzip
import copy
import numbers
import multiprocessing
import json as json
# from lsst.utils import getPackageDir
# from lsst.sims.catalogs.decorators import register_method, compound
# from lsst.sims.photUtils import Sed, BandpassDict
# from lsst.sims.utils.CodeUtilities import sims_clean_up
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

import time

class Variability(object):
    """
    Variability class for adding temporal variation to the magnitudes of
    objects in the base catalog.

    This class provides methods that all variability models rely on.
    Actual implementations of variability models will be provided by
    the *VariabilityModels classes.
    """

    _survey_start = 59580.0 # start time of the LSST survey being simulated (MJD)

    variabilityInitialized = False

    def num_variable_obj(self, params):
        """
        Return the total number of objects in the catalog

        Parameters
        ----------
        params is the dict of parameter arrays passed to a variability method

        Returns
        -------
        The number of objects in the catalog
        """
        params_keys = list(params.keys())
        if len(params_keys) == 0:
            return 0

        return len(params[params_keys[0]])

    def initializeVariability(self, doCache=False):
        """
        It will only be called from applyVariability, and only
        if self.variabilityInitiailized == False (which this method then
        sets to True)

        @param [in] doCache controls whether or not the code caches calculated
        light curves for future use
        """
        # Docstring is a best approximation of what this method does.
        # This is older code.

        self.variabilityInitialized=True
        #below are variables to cache the light curves of variability models
        self.variabilityLcCache = {}
        self.variabilityCache = doCache
        try:
            self.variabilityDataDir = os.environ.get("SIMS_SED_LIBRARY_DIR")
        except:
            raise RuntimeError("sims_sed_library must be setup to compute variability because it contains"+
                               " the lightcurves")



    def applyVariability(self, varParams_arr, expmjd=None,
                         variability_cache=None):
        """
        Read in an array/list of varParamStr objects taken from the CatSim
        database.  For each varParamStr, call the appropriate variability
        model to calculate magnitude offsets that need to be applied to
        the corresponding astrophysical offsets.  Return a 2-D numpy
        array of magnitude offsets in which each row is an LSST band
        in ugrizy order and each column is an astrophysical object from
        the CatSim database.

        variability_cache is a cache of data as initialized by the
        create_variability_cache() method (optional; if None, the
        method will just use a globl cache)
        """
        t_start = time.time()
        if not hasattr(self, '_total_t_apply_var'):
            self._total_t_apply_var = 0.0

        # construct a registry of all of the variability models
        # available to the InstanceCatalog
        if not hasattr(self, '_methodRegistry'):
            self._methodRegistry = {}
            self._method_name_to_int = {}
            next_int = 0
            for methodname in dir(self):
                method=getattr(self, methodname)
                if hasattr(method, '_registryKey'):
                    if method._registryKey not in self._methodRegistry:
                        self._methodRegistry[method._registryKey] = method
                        self._method_name_to_int[method._registryKey] = next_int
                        next_int += 1

        if self.variabilityInitialized == False:
            self.initializeVariability(doCache=True)


        if isinstance(expmjd, numbers.Number) or expmjd is None:
            # A numpy array of magnitude offsets.  Each row is
            # an LSST band in ugrizy order.  Each column is an
            # astrophysical object from the CatSim database.
            deltaMag = np.zeros((6, len(varParams_arr)))
        else:
            # the last dimension varies over time
            deltaMag = np.zeros((6, len(varParams_arr), len(expmjd)))

        # When the InstanceCatalog calls all of its getters
        # with an empty chunk to check column dependencies,
        # call all of the variability models in the
        # _methodRegistry to make sure that all of the column
        # dependencies of the variability models are detected.
        if len(varParams_arr) == 0:
            for method_name in self._methodRegistry:
                self._methodRegistry[method_name]([],{},0)

        # Keep a list of all of the specific variability models
        # that need to be called.  There is one entry for each
        # astrophysical object in the CatSim database.  We will
        # ultimately run np.where on method_name_arr to determine
        # which objects need to be passed through which
        # variability methods.
        method_name_arr = []

        # also keep an array listing the methods to use
        # by the integers mapped with self._method_name_to_int;
        # this is for faster application of np.where when
        # figuring out which objects go with which method
        method_int_arr = -1*np.ones(len(varParams_arr), dtype=int)

        # Keep a dict keyed on all of the method names in
        # method_name_arr.  params[method_name] will be another
        # dict keyed on the names of the parameters required by
        # the method method_name.  The values of this dict will
        # be lists of parameter values for all astrophysical
        # objects in the CatSim database.  Even objects that
        # do no callon method_name will have entries in these
        # lists (they will be set to None).
        params = {}

        for ix, varCmd in enumerate(varParams_arr):
            if str(varCmd) == 'None':
                continue

            varCmd = json.loads(varCmd)

            # find the key associated with the name of
            # the specific variability model to be applied
            if 'varMethodName' in varCmd:
                meth_key = 'varMethodName'
            else:
                meth_key = 'm'

            # find the key associated with the list of
            # parameters to be supplied to the variability
            # model
            if 'pars' in varCmd:
                par_key = 'pars'
            else:
                par_key = 'p'

            # if we have discovered a new variability model
            # that needs to be called, initialize its entries
            # in the params dict
            if varCmd[meth_key] not in method_name_arr:
                params[varCmd[meth_key]] = {}
                for p_name in varCmd[par_key]:
                    params[varCmd[meth_key]][p_name] = [None]*len(varParams_arr)

            method_name_arr.append(varCmd[meth_key])
            if varCmd[meth_key] != 'None':
                try:
                    method_int_arr[ix] = self._method_name_to_int[varCmd[meth_key]]
                except KeyError:
                    raise RuntimeError("Your InstanceCatalog does not contain " \
                                       + "a variability method corresponding to '%s'"
                                       % varCmd[meth_key])

            for p_name in varCmd[par_key]:
                params[varCmd[meth_key]][p_name][ix] = varCmd[par_key][p_name]

        method_name_arr = np.array(method_name_arr)
        for method_name in params:
            for p_name in params[method_name]:
                params[method_name][p_name] = np.array(params[method_name][p_name])

        # Loop over all of the variability models that need to be called.
        # Call each variability model on the astrophysical objects that
        # require the model.  Add the result to deltaMag.
        for method_name in np.unique(method_name_arr):
            if method_name != 'None':

                if expmjd is None:
                    expmjd = self.obs_metadata.mjd.TAI

                deltaMag += self._methodRegistry[method_name](np.where(method_int_arr==self._method_name_to_int[method_name]),
                                                              params[method_name],
                                                              expmjd,
                                                              variability_cache=variability_cache)

        self._total_t_apply_var += time.time()-t_start
        return deltaMag


    def applyStdPeriodic(self, valid_dexes, params, keymap, expmjd,
                         inDays=True, interpFactory=None):

        """
        Applies a specified variability method.

        The params for the method are provided in the dict params{}

        The keys for those parameters are in the dict keymap{}

        This is because the syntax used here is not necessarily the syntax
        used in the data bases.

        The method will return a dict of magnitude offsets.  The dict will
        be keyed to the filter names.

        @param [in] valid_dexes is the result of numpy.where() indicating
        which astrophysical objects from the CatSim database actually use
        this variability model.

        @param [in] params is a dict of parameters for the variability model.
        The dict is keyed to the names of parameters.  The values are arrays
        of parameter values.

        @param [in] keymap is a dict mapping from the parameter naming convention
        used by the database to the parameter naming convention used by the
        variability methods below.

        @param [in] expmjd is the mjd of the observation

        @param [in] inDays controls whether or not the time grid
        of the light curve is renormalized by the period

        @param [in] interpFactory is the method used for interpolating
        the light curve

        @param [out] magoff is a 2D numpy array of magnitude offsets.  Each
        row is an LSST band in ugrizy order.  Each column is a different
        astrophysical object from the CatSim database.
        """
        if isinstance(expmjd, numbers.Number):
            magoff = np.zeros((6, self.num_variable_obj(params)))
        else:
            magoff = np.zeros((6, self.num_variable_obj(params), len(expmjd)))
        expmjd = np.asarray(expmjd)
        for ix in valid_dexes[0]:
            filename = params[keymap['filename']][ix]
            toff = params[keymap['t0']][ix]

            inPeriod = None
            if 'period' in params:
                inPeriod = params['period'][ix]

            epoch = expmjd - toff
            if filename in self.variabilityLcCache:
                splines = self.variabilityLcCache[filename]['splines']
                period = self.variabilityLcCache[filename]['period']
            else:
                lc = np.loadtxt(os.path.join(self.variabilityDataDir,filename), unpack=True, comments='#')
                if inPeriod is None:
                    dt = lc[0][1] - lc[0][0]
                    period = lc[0][-1] + dt
                else:
                    period = inPeriod

                if inDays:
                    lc[0] /= period

                splines  = {}

                if interpFactory is not None:
                    splines['u'] = interpFactory(lc[0], lc[1])
                    splines['g'] = interpFactory(lc[0], lc[2])
                    splines['r'] = interpFactory(lc[0], lc[3])
                    splines['i'] = interpFactory(lc[0], lc[4])
                    splines['z'] = interpFactory(lc[0], lc[5])
                    splines['y'] = interpFactory(lc[0], lc[6])
                    if self.variabilityCache:
                        self.variabilityLcCache[filename] = {'splines':splines, 'period':period}
                else:
                    splines['u'] = interp1d(lc[0], lc[1])
                    splines['g'] = interp1d(lc[0], lc[2])
                    splines['r'] = interp1d(lc[0], lc[3])
                    splines['i'] = interp1d(lc[0], lc[4])
                    splines['z'] = interp1d(lc[0], lc[5])
                    splines['y'] = interp1d(lc[0], lc[6])
                    if self.variabilityCache:
                        self.variabilityLcCache[filename] = {'splines':splines, 'period':period}

            phase = epoch/period - epoch//period
            magoff[0][ix] = splines['u'](phase)
            magoff[1][ix] = splines['g'](phase)
            magoff[2][ix] = splines['r'](phase)
            magoff[3][ix] = splines['i'](phase)
            magoff[4][ix] = splines['z'](phase)
            magoff[5][ix] = splines['y'](phase)

        return magoff


class ExtraGalacticVariabilityModels(Variability):
    """
    A mixin providing the model for AGN variability.
    """

    _agn_walk_start_date = 58580.0
    _agn_threads = 1

    @register_method('applyAgn')
    def applyAgn(self, valid_dexes, params, expmjd,
                 variability_cache=None, redshift=None):

        if redshift is None:
            redshift_arr = self.column_by_name('redshift')
        else:
            redshift_arr = redshift

        if len(params) == 0:
            return np.array([[],[],[],[],[],[]])

        if isinstance(expmjd, numbers.Number):
            dMags = np.zeros((6, self.num_variable_obj(params)))
            max_mjd = expmjd
            min_mjd = expmjd
            mjd_is_number = True
        else:
            dMags = np.zeros((6, self.num_variable_obj(params), len(expmjd)))
            max_mjd = max(expmjd)
            min_mjd = min(expmjd)
            mjd_is_number = False

        seed_arr = params['seed']
        tau_arr = params['agn_tau'].astype(float)
        sfu_arr = params['agn_sfu'].astype(float)
        sfg_arr = params['agn_sfg'].astype(float)
        sfr_arr = params['agn_sfr'].astype(float)
        sfi_arr = params['agn_sfi'].astype(float)
        sfz_arr = params['agn_sfz'].astype(float)
        sfy_arr = params['agn_sfy'].astype(float)

        duration_observer_frame = max_mjd - self._agn_walk_start_date

        if duration_observer_frame < 0 or min_mjd < self._agn_walk_start_date:
            raise RuntimeError("WARNING: Time offset greater than minimum epoch.  " +
                               "Not applying variability. "+
                               "expmjd: %e should be > start_date: %e  " % (min_mjd, self._agn_walk_start_date) +
                               "in applyAgn variability method")

        if self._agn_threads == 1 or len(valid_dexes[0])==1:
            for i_obj in valid_dexes[0]:
                seed = seed_arr[i_obj]
                tau = tau_arr[i_obj]
                time_dilation = 1.0+redshift_arr[i_obj]
                sf_u = sfu_arr[i_obj]
                dMags[0][i_obj] = self._simulate_agn(expmjd, tau, time_dilation, sf_u, seed)
        # else:
        #     p_list = []

        #     mgr = multiprocessing.Manager()
        #     if mjd_is_number:
        #         out_struct = mgr.Array('d', [0]*len(valid_dexes[0]))
        #     else:
        #         out_struct = mgr.dict()

        #     #################
        #     # Try to subdivide the AGN into batches such that the number
        #     # of time steps simulated by each thread is close to equal
        #     tot_steps = 0
        #     n_steps = []
        #     for tt, zz in zip(tau_arr[valid_dexes], redshift_arr[valid_dexes]):
        #         dilation = 1.0+zz
        #         dt = tt/100.0
        #         dur = (duration_observer_frame/dilation)
        #         nt = dur/dt
        #         tot_steps += nt
        #         n_steps.append(nt)

        #     batch_target = tot_steps/self._agn_threads
        #     i_start_arr = [0]
        #     i_end_arr = []
        #     current_batch = n_steps[0]
        #     for ii in range(1,len(n_steps),1):
        #         current_batch += n_steps[ii]
        #         if ii == len(n_steps)-1:
        #             i_end_arr.append(len(n_steps))
        #         elif len(i_start_arr)<self._agn_threads:
        #             if current_batch>=batch_target:
        #                 i_end_arr.append(ii)
        #                 i_start_arr.append(ii)
        #                 current_batch = n_steps[ii]

        #     if len(i_start_arr) != len(i_end_arr):
        #         raise RuntimeError('len i_start %d len i_end %d; dexes %d' %
        #                            (len(i_start_arr),
        #                             len(i_end_arr),
        #                             len(valid_dexes[0])))
        #     assert len(i_start_arr) <= self._agn_threads
        #     ############

        #     # Actually simulate the AGN on the the number of threads allotted
        #     for i_start, i_end in zip(i_start_arr, i_end_arr):
        #         dexes = valid_dexes[0][i_start:i_end]
        #         if mjd_is_number:
        #             out_dexes = range(i_start,i_end,1)
        #         else:
        #             out_dexes = dexes
        #         p = multiprocessing.Process(target=self._threaded_simulate_agn,
        #                                     args=(expmjd, tau_arr[dexes],
        #                                           1.0+redshift_arr[dexes],
        #                                           sfu_arr[dexes],
        #                                           seed_arr[dexes],
        #                                           out_dexes,
        #                                           out_struct))
        #         p.start()
        #         p_list.append(p)
        #     for p in p_list:
        #         p.join()

        #     if mjd_is_number:
        #         dMags[0][valid_dexes] = out_struct[:]
        #     else:
        #         for i_obj in out_struct.keys():
        #             dMags[0][i_obj] = out_struct[i_obj]

        for i_filter, filter_name in enumerate(('g', 'r', 'i', 'z', 'y')):
            for i_obj in valid_dexes[0]:
                dMags[i_filter+1][i_obj] = dMags[0][i_obj]*params['agn_sf%s' % filter_name][i_obj]/params['agn_sfu'][i_obj]

        return dMags

    # def _threaded_simulate_agn(self, expmjd, tau_arr,
    #                            time_dilation_arr, sf_u_arr,
    #                            seed_arr, dex_arr, out_struct):

    #     if isinstance(expmjd, numbers.Number):
    #         mjd_is_number = True
    #     else:
    #         mjd_is_number = False

    #     for tau, time_dilation, sf_u, seed, dex in zip(tau_arr, time_dilation_arr, sf_u_arr, seed_arr, dex_arr):
    #         out_struct[dex] = self._simulate_agn(expmjd, tau, time_dilation,
    #                                              sf_u, seed)

    def _simulate_agn(self, expmjd, tau, time_dilation, sf_u, seed):
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
                duration_observer_frame = max(expmjd) - self._agn_walk_start_date
            else:
                duration_observer_frame = expmjd - self._agn_walk_start_date


            rng = np.random.RandomState(seed)
            dt = tau/100.
            duration_rest_frame = duration_observer_frame/time_dilation
            nbins = int(math.ceil(duration_rest_frame/dt))+1

            time_dexes = np.round((expmjd-self._agn_walk_start_date)/(time_dilation*dt)).astype(int)
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
                        dm_val = ((expmjd-self._agn_walk_start_date)*(dx1-dx2)/time_dilation+dx2*x1-dx1*x2)/(x1-x2)
                        d_m_out = dm_val
                    else:
                        for i_time_out in time_dex_map[i_time]:
                            local_end = (expmjd[i_time_out]-self._agn_walk_start_date)/time_dilation
                            dm_val = (local_end*(dx1-dx2)+dx2*x1-dx1*x2)/(x1-x2)
                            d_m_out[i_time_out] = dm_val

            return d_m_out


def test(sf, u, rest_frame):
    

