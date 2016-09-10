'''
This module contains helper functions for generating rampedpyrox summary tables.
'''


from __future__ import print_function

__docformat__ = 'restructuredtext en'
__all__ = ['_timedata_peak_info', '_energycomplex_peak_info',
			'_rpo_isotopes_frac_info', '_rpo_isotopes_peak_info']

import numpy as np
import pandas as pd

#import exceptions
from .exceptions import(
	RunModelError,
	)

#define function to calculate timedata peak info and store
def _timedata_peak_info(timedata):
	'''
	Calculates the ``rp.TimeData`` instance peak info and stores as a 
	DataFrame.

	Parameters
	----------
	timedata : rp.TimeData
		``rp.TimeData`` instance containing peaks to be summarized.

	Returns
	-------
	peak_info : pd.DataFrame
		DataFrame of resulting peak info.

	Raises
	------
	RunModelError
		If TimeData instance does not contain necessary attributes (i.e. if it
		does not have inputted model-estimated data).
	'''

	#raise exception if timedata doesn't contain peaks
	if not hasattr(timedata, 'gam'):
		raise RunModelError(
			'TimeData instance contains no model-fitted data! Run forward'
			'model before trying to summarize peaks.')

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

	peak_info = np.column_stack(
		(t_max, T_max, height_t, height_T, rel_area))
	
	cols = ['t max (s)', 'T max (K)', 'max rate (frac/s)', 
		'max rate (frac/K)', 'rel. area']

	peak_info = pd.DataFrame(
		peak_info, 
		columns = cols,
		index = np.arange(1, timedata.nPeak + 1))

	return peak_info

def _energycomplex_peak_info(ratedata):
	'''
	Calculates the ``rp.EnergyComplex`` instance peak info and stores as a 
	DataFrame.

	Parameters
	----------
	ratedata : rp.EnergyComplex
		``rp.EnergyComplex`` instance containing peaks to be summarized.

	Returns
	-------
	peak_info : pd.DataFrame
		DataFrame of resulting peak info.

	Raises
	------
	RunModelError
		If EnergyComplex instance does not contain necessary attributes (i.e. if it
		does not have inputted model-estimated data).
	'''

	#raise exception if timedata doesn't contain peaks
	if not hasattr(ratedata, 'peaks'):
		raise RunModelError(
			'RateData instance contains no model-fitted data! Run inverse'
			'model before trying to summarize peaks.')

	#set pandas display options
	pd.set_option('precision', 2)		

	#calculate relative area and append to peak_info
	rel_area = np.sum(ratedata.peaks, axis = 0)/np.sum(ratedata.peaks)
	peak_info = np.column_stack((ratedata._pkinf, rel_area))

	#combine peak_info into pandas dataframe

	cols = ['mu (kJ/mol)', 'sigma (kJ/mol)', 'height', 'rel. area']

	peak_info = pd.DataFrame(
		peak_info, 
		columns = cols,
		index = np.arange(1, ratedata.nPeak + 1))

	return peak_info

def _rpo_isotopes_frac_info(rpoisotopes):
	'''
	Calculates the ``rp.RpoIsotopes`` instance fraction info and stores as a
	DataFrame.

	Parameters
	----------
	rpoisotopes : rp.RpoIsotopes
		``rp.RpoIsotopes`` instance containing fractions to be summarized.

	Returns
	-------
	frac_info : pd.DataFrame
		DataFrame of resulting fraction info.
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
	frac_info = pd.DataFrame(
		info,
		columns = names,
		index = np.arange(1, rpoisotopes.nFrac + 1))

	return frac_info

def _rpo_isotopes_peak_info(cmbd, DEa, rpoisotopes):
	'''
	Calculates the ``rp.RpoIsotopes`` instance peak info and stores as a
	DataFrame.

	Parameters
	----------
	cmbd : list
		The fractions that have been combined. Will use to repeat info.

	DEa : np.ndarray
		Array of the DEa used for the KIE in each peak.

	rpoisotopes : rp.RpoIsotopes
		``rp.RpoIsotopes`` instance containing peaks to be summarized.

	Returns
	-------
	peak_info : pd.DataFrame
		DataFrame instance of resulting fraction info.
	'''

	#create empty list to store existing data
	info = []

	#create empty list to store name strings
	names = []

	#go through each measurement and add if it exists

	#peak mass
	if hasattr(rpoisotopes, 'm_peak'):

		#extract values
		m_peak = rpoisotopes.m_peak
		m_peak_std = rpoisotopes.m_peak_std

		#keep track of the rows to add back in if cmbd
		if cmbd is not None:

			#calculate indices of deleted peaks
			dp = [val - i for i, val in enumerate(cmbd)]
			dp = np.array(dp) #convert to nparray
			
			#insert deleted peaks back in
			m_peak = np.insert(
				m_peak, 
				dp, 
				m_peak[dp-1])

			m_peak_std = np.insert(
				m_peak_std, 
				dp, 
				m_peak_std[dp-1])

		#append lists with data
		info.append(m_peak)
		info.append(m_peak_std)
		
		names.append('mass (ugC)')
		names.append('mass std. (ugC)')

	#peak d13C
	if hasattr(rpoisotopes, 'd13C_peak'):

		#extract values
		d13C_peak = rpoisotopes.d13C_peak
		d13C_peak_std = rpoisotopes.d13C_peak_std

		#keep track of the rows to add back in if cmbd
		if cmbd is not None:

			#calculate indices of deleted peaks
			dp = [val - i for i, val in enumerate(cmbd)]
			dp = np.array(dp) #convert to nparray
			
			#insert deleted peaks back in
			d13C_peak = np.insert(
				d13C_peak, 
				dp, 
				d13C_peak[dp-1])

			d13C_peak_std = np.insert(
				d13C_peak_std, 
				dp, 
				d13C_peak_std[dp-1])

		#append lists with data
		info.append(d13C_peak)
		info.append(d13C_peak_std)
		
		names.append('d13C (VPDB)')
		names.append('d13C std. (VPDB)')

	#peak Fm
	if hasattr(rpoisotopes, 'Fm_peak'):

		#extract values
		Fm_peak = rpoisotopes.Fm_peak
		Fm_peak_std = rpoisotopes.Fm_peak_std

		#keep track of the rows to add back in if cmbd
		if cmbd is not None:

			#calculate indices of deleted peaks
			dp = [val - i for i, val in enumerate(cmbd)]
			dp = np.array(dp) #convert to nparray
			
			#insert deleted peaks back in
			Fm_peak = np.insert(
				Fm_peak, 
				dp, 
				Fm_peak[dp-1])

			Fm_peak_std = np.insert(
				Fm_peak_std, 
				dp, 
				Fm_peak_std[dp-1])

		#append lists with data
		info.append(Fm_peak)
		info.append(Fm_peak_std)
		
		names.append('Fm')
		names.append('Fm std.')

	#add DEa info
	info.append(DEa)
	names.append('DEa (kJ/mol)')

	info = np.column_stack(info)
	
	#generate index with asterisks by combined peaks
	index = range(info.shape[0])
	istr =  ["{:1d}".format(x+1) for x in index]

	if cmbd is not None:
		istr = [val+'*' if i in cmbd else val for i, val in enumerate(istr)]

	#set pandas display options
	pd.set_option('precision', 2)

	#store in dataframe
	peak_info = pd.DataFrame(
		info,
		columns = names,
		index = istr)

	return peak_info
