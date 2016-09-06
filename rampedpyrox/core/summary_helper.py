'''
This module contains helper functions for generating rampedpyrox summary tables.
'''

import numpy as np
import pandas as pd


#define function to calculate timedata peak info and store
def _timedata_peak_info(timedata):
	'''
	Calculates the ``TimeData`` instance peak info and stores as a 
	``pd.DataFrame``.

	Parameters
	----------
	timedata : rp.TimeData
		TimeData instance containing peaks to be summarized.

	Returns
	-------
	peak_info : pd.DataFrame
		DataFrame instance of resulting peak info.

	Raises
	------
	AttributeError
		If TimeData instance does not contain necessary attributes (i.e. if it
		does not have inputted model-estimated data).
	'''

	#raise exception if timedata doesn't contain peaks
	if not hasattr(timedata, 'gam'):
		raise AttributeError((
			"TimeData instance contains no model-fitted data! Run forward"
			"model before trying to summarize peaks."))

	#set pandas display options
	pd.set_option('precision', 2)

	#calculate peak indices
	i = np.argmax(-timedata.dcmptdt, axis=0)

	#extract info at peaks
	t_max = timedata.t[i]
	T_max = timedata.T[i]
	height_t = np.diag(-timedata.dcmptdt[i])
	height_T = np.diag(-timedata.dcmptdT[i])
	rel_area = timedata.cmpt[0,:]

	peak_info = np.column_stack((t_max, T_max, height_t, height_T, rel_area))
	peak_info = pd.DataFrame(peak_info, 
		columns = ['t max (s)', 'T max (K)', 'max rate (frac/s)', \
			'max rate (frac/K)','rel. area'],
		index = np.arange(1, timedata.nPeak + 1))

	return peak_info

def _energycomplex_peak_info(ratedata, peak_info):
	'''
	Calculates the ``EnergyComplex`` instance peak info and stores as a 
	``pd.DataFrame``.

	Parameters
	----------
	ratedata : rp.EnergyComplex
		EnergyComplex instance containing peaks to be summarized.

	peak_info : np.ndarray
		2d array of peak mu, sigma, and height

	Returns
	-------
	peak_info : pd.DataFrame
		DataFrame instance of resulting peak info.

	Raises
	------
	AttributeError
		If EnergyComplex instance does not contain necessary attributes (i.e. if it
		does not have inputted model-estimated data).
	'''

	#raise exception if timedata doesn't contain peaks
	if not hasattr(ratedata, 'peaks'):
		raise AttributeError((
			"RateData instance contains no model-fitted data! Run inverse"
			"model before trying to summarize peaks."))

	#set pandas display options
	pd.set_option('precision', 2)		

	#calculate relative area and append to peak_info
	rel_area = np.sum(ratedata.peaks, axis = 0)/np.sum(ratedata.peaks)
	peak_info = np.column_stack((peak_info, rel_area))

	#combine peak_info into pandas dataframe
	peak_info = pd.DataFrame(peak_info, 
		columns = ['mu (kJ)', 'sigma (kJ)', 'height', 'rel. area'],
		index = np.arange(1, ratedata.nPeak + 1))

	return peak_info

def _rpo_isotopes_frac_info(rpoisotopes):
	'''
	Calculates the ``RpoIsotopes`` instance fraction info and stores as a
	``pd.DataFrame`` instance.

	Parameters
	----------
	rpoisotopes : rp.RpoIsotopes
		RpoIsotopes instance containing fractions to be summarized.

	Returns
	-------
	frac_info : pd.DataFrame
		DataFrame instance of resulting fraction info.
	'''

	#create empty list to store existing data
	info = []

	#create empty list to store name strings
	names = []

	#add t_frac
	info.append(rpoisotopes.t_frac[:,0])
	info.append(rpoisotopes.t_frac[:,1])
	names.append('t0 (s)')
	names.append('tf (s)')

	#go through each measurement and add if it exists
	if hasattr(rpoisotopes, 'm_frac'):
		info.append(rpoisotopes.m_frac)
		info.append(rpoisotopes.m_frac_std)
		names.append('mass (ugC)')
		names.append('mass std. (ugC)')

	if hasattr(rpoisotopes, 'd13C_frac'):
		info.append(rpoisotopes.d13C_frac)
		info.append(rpoisotopes.d13C_frac_std)
		names.append('d13C (VPDB)')
		names.append('d13C std. (VPDB)')

	if hasattr(rpoisotopes, 'Fm_frac'):
		info.append(rpoisotopes.Fm_frac)
		info.append(rpoisotopes.Fm_frac_std)
		names.append('Fm')
		names.append('Fm std.')

	info = np.column_stack(info)
	
	#set pandas display options
	pd.set_option('precision', 2)

	#store in dataframe
	frac_info = pd.DataFrame(info,
		columns = names,
		index = np.arange(1, rpoisotopes.nFrac + 1))

	return frac_info



















