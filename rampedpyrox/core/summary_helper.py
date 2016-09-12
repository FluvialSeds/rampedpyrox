'''
This module contains helper functions for generating rampedpyrox summary tables.
'''


from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_timedata_peak_info', '_energycomplex_peak_info',
			'_rpo_isotopes_frac_info', '_rpo_isotopes_peak_info']

import numpy as np
import pandas as pd

#import exceptions
from .exceptions import(
	RunModelError,
	)

#define function to calculate timedata component info and store
def _timedata_cmpt_info(timedata):
	'''
	Calculates the ``rp.TimeData`` instance component info and stores as a 
	DataFrame.

	Parameters
	----------
	timedata : rp.TimeData
		``rp.TimeData`` instance containing components to be summarized.

	Returns
	-------
	cmpt_info : pd.DataFrame
		DataFrame of resulting component info.

	Raises
	------
	RunModelError
		If TimeData instance does not contain necessary attributes (i.e. if it
		does not have inputted model-estimated data).
	'''

	#raise exception if timedata doesn't contain components
	if not hasattr(timedata, 'gam'):
		raise RunModelError(
			'TimeData instance contains no model-fitted data! Run forward'
			'model before trying to summarize components.')

	#set pandas display options
	pd.set_option('precision', 2)

	#calculate component indices
	i = np.argmax(-timedata.dcmptdt, axis=0)

	#extract info at component
	t_max = timedata.t[i]
	T_max = timedata.T[i]
	height_t = np.diag(-timedata.dcmptdt[i])
	height_T = np.diag(-timedata.dcmptdT[i])
	rel_area = timedata.cmpt[0,:]

	cmpt_info = np.column_stack(
		(t_max, T_max, height_t, height_T, rel_area))
	
	cols = ['t max (s)', 'T max (K)', 'max rate (frac/s)', 
		'max rate (frac/K)', 'rel. area']

	cmpt_info = pd.DataFrame(
		cmpt_info, 
		columns = cols,
		index = np.arange(1, timedata.nCmpt + 1))

	return cmpt_info

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

def _rpo_isotopes_cmpt_info(cmbd, DEa, rpoisotopes):
	'''
	Calculates the ``rp.RpoIsotopes`` instance component info and stores as a
	DataFrame.

	Parameters
	----------
	cmbd : list
		The fractions that have been combined. Will use to repeat info.

	DEa : np.ndarray
		Array of the DEa used for the KIE in each peak.

	rpoisotopes : rp.RpoIsotopes
		``rp.RpoIsotopes`` instance containing components to be summarized.

	Returns
	-------
	cmpt_info : pd.DataFrame
		DataFrame instance of resulting fraction info.
	'''

	#create empty list to store existing data
	info = []

	#create empty list to store name strings
	names = []

	#go through each measurement and add if it exists

	#component mass
	if hasattr(rpoisotopes, 'm_cmpt'):

		#extract values
		m_cmpt = rpoisotopes.m_cmpt
		m_cmpt_std = rpoisotopes.m_cmpt_std

		#keep track of the rows to add back in if cmbd
		if cmbd is not None:

			#calculate indices of deleted peaks
			dp = [val - i for i, val in enumerate(cmbd)]
			dp = np.array(dp) #convert to nparray
			
			#insert deleted peaks back in
			m_cmpt = np.insert(
				m_cmpt, 
				dp, 
				m_cmpt[dp-1])

			m_cmpt_std = np.insert(
				m_cmpt_std, 
				dp, 
				m_cmpt_std[dp-1])

		#append lists with data
		info.append(m_cmpt)
		info.append(m_cmpt_std)
		
		names.append('mass (ugC)')
		names.append('mass std. (ugC)')

	#component d13C
	if hasattr(rpoisotopes, 'd13C_cmpt'):

		#extract values
		d13C_cmpt = rpoisotopes.d13C_cmpt
		d13C_cmpt_std = rpoisotopes.d13C_cmpt_std

		#keep track of the rows to add back in if cmbd
		if cmbd is not None:

			#calculate indices of deleted peaks
			dp = [val - i for i, val in enumerate(cmbd)]
			dp = np.array(dp) #convert to nparray
			
			#insert deleted peaks back in
			d13C_cmpt = np.insert(
				d13C_cmpt, 
				dp, 
				d13C_cmpt[dp-1])

			d13C_cmpt_std = np.insert(
				d13C_cmpt_std, 
				dp, 
				d13C_cmpt_std[dp-1])

		#append lists with data
		info.append(d13C_cmpt)
		info.append(d13C_cmpt_std)
		
		names.append('d13C (VPDB)')
		names.append('d13C std. (VPDB)')

	#component Fm
	if hasattr(rpoisotopes, 'Fm_cmpt'):

		#extract values
		Fm_cmpt = rpoisotopes.Fm_cmpt
		Fm_cmpt_std = rpoisotopes.Fm_cmpt_std

		#keep track of the rows to add back in if cmbd
		if cmbd is not None:

			#calculate indices of deleted peaks
			dp = [val - i for i, val in enumerate(cmbd)]
			dp = np.array(dp) #convert to nparray
			
			#insert deleted peaks back in
			Fm_cmpt = np.insert(
				Fm_cmpt, 
				dp, 
				Fm_cmpt[dp-1])

			Fm_cmpt_std = np.insert(
				Fm_cmpt_std, 
				dp, 
				Fm_cmpt_std[dp-1])

		#append lists with data
		info.append(Fm_cmpt)
		info.append(Fm_cmpt_std)
		
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
	cmpt_info = pd.DataFrame(
		info,
		columns = names,
		index = istr)

	return cmpt_info
