'''
This module contains helper functions for generating rampedpyrox summary tables.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_calc_rate_info', '_calc_RPO_info', '_rpo_isotopes_frac_info']

import numpy as np
import pandas as pd

#import helper functions
from .core_functions import(
	derivatize,
	extract_moments
	)

#define function to calculate ratedata info and store
def _calc_rate_info(k, p, kstr = 'E'):
	'''
	Calculates the ``rp.RateData`` instance summary statistics and stores in
	a Series.

	Parameters
	----------
	k : np.ndarray
		Array of rates (or E), length `nk`.

	p : np.ndarray
		Array of pdf of rates (or E), length `nk`.

	kstr : string
		String of nomenclature for k (i.e. 'k' or 'E')

	Returns
	-------
	rate_summary : pd.Series
		Series of resulting RateData summary info.
	'''

	#set pandas display options
	pd.set_option('precision', 2)

	if kstr == 'E':
		unit = ' (kJ/mol)'

	else:
		unit = ' (s-1)'

	#define series index
	ind = [kstr + '_max' + unit,
			kstr + '_mean' + unit,
			kstr + '_std' + unit,
			'p0(' + kstr + ')_max']

	#find max
	i = np.where(p == np.max(p))[0][0]

	#calculate statistics
	kmax = k[i]
	pmax = p[i]
	kav, kstd = extract_moments(k, p)

	#combine into list
	vals = [kmax, kav, kstd, pmax]

	#make series
	rate_summary = pd.Series(vals, index = ind)

	return rate_summary

#define function to calculate timedata info and store
def _calc_RPO_info(t, T, g):
	'''
	Calculates the ``rp.TimeData`` instance thermogram summary statistics and
	stores in a Series.

	Parameters
	----------
	t : numpy.ndarray
		Array of timepoints, in seconds. Length `nt`.

	T : numpy.ndarray
		Array of temperature, in Kelvin. Length `nt`.

	g : numpy.ndarray
		Array of the true fraction of carbon remaining at each timepoint.
		Length `nt`.

	Returns
	-------
	tg_summary : pd.Series
		Series of resulting thermogram summary info.
	'''

	#set pandas display options
	pd.set_option('precision', 2)

	#define series index
	ind = ['t_max (s)',
			't_mean (s)',
			't_std (s)',
			'T_max (K)',
			'T_mean (K)',
			'T_std (K)',
			'max_rate (frac/s)',
			'max_rate (frac/K)']

	#derivatize g
	dgdt = derivatize(g,t)
	dgdT = derivatize(g,T)

	#find max
	i = np.where(dgdt == np.min(dgdt))[0][0]

	#calculate statistics
	tmax = t[i]
	Tmax = T[i]
	rmax_s = -dgdt[i]
	rmax_k = -dgdT[i]
	tav, tstd = extract_moments(t, dgdt)
	Tav, Tstd = extract_moments(T, dgdT)

	#combine into list
	vals = [tmax, tav, tstd, Tmax, Tav, Tstd, rmax_s, rmax_k]

	#make series
	tg_summary = pd.Series(vals, index = ind)

	return tg_summary

#create method for calculating RPO isotope summary info table
def _calc_ri_info(ri, flag = 'raw'):
	'''
	Calculates the ``rp.RpoIsotopes`` instance fraction info and stores as a
	DataFrame.

	Parameters
	----------
	ri : rp.RpoIsotopes
		``rp.RpoIsotopes`` instance containing fractions to be summarized.

	flag : str
		Tells the method whether to store raw or corrected data

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
	info.append(ri.t_frac[:,0])
	info.append(ri.t_frac[:,1])
	names.append('t0 (s)')
	names.append('tf (s)')

	#add E
	info.append(ri.E_frac)
	info.append(ri.E_frac_std)
	names.append('E (kJ/mol)')
	names.append('E std. (kJ/mol)')

	#go through each measurement and add if it exists
	m = 'm_' + flag
	m_std = 'm_' + flag + '_std'

	if hasattr(ri, m):
		
		info.append(getattr(ri, m))
		info.append(getattr(ri, m_std))
		
		names.append('mass (ugC)')
		names.append('mass std. (ugC)')

	d13C = 'd13C_' + flag
	d13C_std = 'd13C_' + flag + '_std'

	if hasattr(ri, d13C):
		
		info.append(getattr(ri, d13C))
		info.append(getattr(ri, d13C_std))
		
		names.append('d13C (VPDB)')
		names.append('d13C std. (VPDB)')

	Fm = 'Fm_' + flag
	Fm_std = 'Fm_' + flag + '_std'

	if hasattr(ri, Fm):
		
		info.append(getattr(ri, Fm))
		info.append(getattr(ri, Fm_std))
		
		names.append('Fm')
		names.append('Fm std.')

	info = np.column_stack(info)
	
	#set pandas display options
	pd.set_option('precision', 2)

	#store in dataframe
	frac_info = pd.DataFrame(
		info,
		columns = names,
		index = np.arange(1, ri.nFrac + 1))

	return frac_info

