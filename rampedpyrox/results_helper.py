'''
This module contains helper functions for the Results class.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_calc_cutoff', '_calc_E_frac','_rpo_blk_corr',
	'_rpo_extract_iso', '_rpo_kie_corr', '_rpo_mass_bal_corr']

import numpy as np
import pandas as pd
import warnings

from collections import Sequence
from numpy.linalg import norm
from scipy.optimize import least_squares
from scipy.optimize import nnls

#import Daem
from .model import(
	Daem
	)

#import exceptions
from .exceptions import(
	ArrayError,
	FileError,
	LengthError,
	ScalarError,
	)

#import helper functions
from .core_functions import(
	assert_len,
	extract_moments,
	)

#define a function to calculate cutoff indices for each RPO fraction
def _calc_cutoff(result, model):
	'''
	Calculates the time index corresponding to the start and stop of each RPO
	fraction. Used for calculating E_frac.

	Parameters
	----------
	result : rp.RpoIsotopes
		``RpoIsotopes`` instance containing CO2 fraction information.

	model : rp.Model
		``rp.Model`` instance containing the A matrix to use for inversion.

	Returns
	-------
	ind_min : np.ndarray
		Index in ``timedata.t`` corresponding to the minimum time for each 
		fraction. Length nFrac.

	ind_max : np.ndarray
		Index in ``timedata.t`` corresponding to the maximum time for each 
		fraction. Length nFrac.
	'''

	#extract shapes
	nFrac = result.nFrac
	nt = model.nt
	
	#extract arrays
	t_frac = result.t_frac
	t = model.t

	#pre-allocate index arrays
	ind_min = np.zeros(nFrac, dtype = int)
	ind_max = np.zeros(nFrac, dtype = int)

	#loop through and calculate indices
	for i, row in enumerate(t_frac):

		#extract indices for each fraction
		ind = np.where((t > row[0]) & (t <= row[1]))[0]

		#store first and last indices
		ind_min[i] = ind[0] - 1 #subtract one so theres no gap!
		ind_max[i] = ind[-1]

	return ind_min, ind_max

#define function to calculate the E values of each RPO fraction
def _calc_E_frac(result, model, ratedata):
	'''
	Method for determining the distribution of E values contained within
	each RPO fraction. Used for calculating `E_frac` and `E_frac_std` in
	order to compare with measured isotope values.

	Parameters
	----------
	result : rp.RpoIsotopes
		``rp.RpoIsotopes`` instance containing the `t_frac` array to be used
		for calculating E in each fraction.

	model : rp.Model
		``rp.Model`` instance containing the A matrix to use for 
		inversion.

	ratedata : rp.RateData
		``rp.Ratedata`` instance containing the reactive continuum data.

	Returns
	-------
	E_frac : np.ndarray
		Array of mean E value contained in each RPO fraction, length `nFrac`.

	E_frac_std : np.ndarray
		Array of the standard deviation of E contained in each RPO fraction,
		length `nFrac`.

	p_frac : np.ndarray
		2d array of the distribution of E contained in each RPO fraction,
		shape [`nFrac` x `nE`].
	'''

	#calculate cutoff indices
	ind_min, ind_max = _calc_cutoff(result, model)

	#extract necessary data
	A = model.A
	E = ratedata.E
	dE = np.gradient(E)
	nE = ratedata.nE
	nF = result.nFrac
	p = ratedata.p

	#make an empty matrix to store results
	p_frac = np.zeros([nF, nE])
	E_frac = np.zeros(nF)
	E_frac_std = np.zeros(nF)

	#loop through each time window and calculate p0E_diff distribution
	for i in range(nF):

		#extract indices
		imin = ind_min[i]
		imax = ind_max[i]

		#p(E,t) at time 0
		pt0 = p*A[imin,:]/dE #divide by dE so total area == 1

		#p(E,t) at time final
		ptf = p*A[imax,:]/dE #divide by dE so total area == 1

		#difference -- i.e. p(E) evolved over Dt
		Dpt = pt0 - ptf

		#store in matrix
		p_frac[i, :] = Dpt

		#calculate the mean and stdev
		E_frac[i], E_frac_std[i] = extract_moments(E, Dpt)

	return E_frac, E_frac_std, p_frac

#define a function to blank-correct fraction isotopes
def _rpo_blk_corr(
		d13C, 
		d13C_std, 
		Fm, 
		Fm_std, 
		m, 
		m_std, 
		t,
		blk_d13C = (-29.0, 0.1),
		blk_flux = (0.375, 0.0583),
		blk_Fm =  (0.555, 0.042)):
	'''
	Performs blank correction on raw isotope values.

	Parameters
	----------
	d13C : None or np.ndarray
		Array of d13C values for each fraction, length nFrac.
	
	d13C_std : np.ndarray
		Array of d13C stdev. for each fraction, length nFrac.

	Fm : None or np.ndarray
		Array of Fm values for each fraction, length nFrac.

	Fm_std : np.ndarray
		Array of Fm stdev. for each fraction, length nFrac.

	m : np.ndarray
		Array of masses (ugC) for each fraction, length nFrac.

	m_std : np.ndarray
		Array of mass stdev. (ugC) for each fraction, length nFrac.

	t : np.ndarray
		2d array of time for each fraction (in seconds), length nFrac.

	blk_d13C : tuple
		Tuple of the blank d13C composition (VPDB), in the form 
		(mean, stdev.) to be used of ``blk_corr = True``. Defaults to the
		NOSAMS RPO blank as calculated by Hemingway et al. **2017**.

	blk_flux : tuple
		Tuple of the blank flux (ng/s), in the form (mean, stdev.) to
		be used of ``blk_corr = True``. Defaults to the NOSAMS RPO blank 
		as calculated by Hemingway et al. **2017**.

	blk_Fm : tuple
		Tuple of the blank Fm value, in the form (mean, stdev.) to
		be used of ``blk_corr = True``. Defaults to the NOSAMS RPO blank 
		as calculated by Hemingway et al. **2017**.

	Returns
	-------
	d13C_corr : None or np.ndarray
		Array of corrected d13C values for each fraction, length nFrac.
	
	d13C_std_corr : np.ndarray
		Array of corrected d13C stdev. for each fraction, length nFrac.

	Fm_corr : None or np.ndarray
		Array of corrected Fm values for each fraction, length nFrac.

	Fm_std_corr : np.ndarray
		Array of corrected Fm stdev. for each fraction, length nFrac.

	m_corr : np.ndarray
		Array of corrected masses (ugC) for each fraction, length nFrac.

	m_std_corr : np.ndarray
		Array of corrected mass stdev. (ugC) for each fraction, length nFrac.
	
	References
	----------
	[1] J.D. Hemingway et al. (2017) Assessing the blank carbon contribution, 
		isotope mass balance, and kinetic isotope fractionation of the ramped 
		pyrolysis/oxidation instrument at NOSAMS. **Radiocarbon**
	'''

	#define constants
	bl_flux = blk_flux[0]/1000 #make ug/s
	bl_flux_std = blk_flux[1]/1000 #make ug/s

	bl_d13C = blk_d13C[0]
	bl_d13C_std = blk_d13C[1]

	bl_Fm = blk_Fm[0]
	bl_Fm_std = blk_Fm[1]

	#calculate blank mass for each fraction
	dt = t[:,1] - t[:,0]
	bl_mass = bl_flux*dt #ug

	#perform blank correction

	#correct mass
	m_corr = m - bl_mass
	m_std_corr = norm(
		[m_std, dt*bl_flux_std], 
		axis = 0)

	#correct d13C
	if d13C is not None:

		dt1 = d13C_std
		dt2 = dt*bl_d13C*bl_flux_std/m
		dt3 = bl_mass*bl_d13C_std/m
		dt4 = bl_mass*bl_d13C*m_std_corr/(m_corr**2)
	
		d13C_corr = (m*d13C - bl_mass*bl_d13C)/m_corr
		d13C_std_corr = norm(
			[dt1, dt2, dt3, dt4], 
			axis = 0)

	else:
		d13C_corr = None
		d13C_corr_std = 0

	#correct Fm
	if Fm is not None:

		ft1 = Fm_std
		ft2 = dt*bl_Fm*bl_flux_std/m
		ft3 = bl_mass*bl_Fm_std/m
		ft4 = bl_mass*bl_Fm*m_std_corr/(m_corr**2)
		
		Fm_corr = (m*Fm - bl_mass*bl_Fm)/m_corr
		Fm_std_corr = norm(
			[ft1, ft2, ft3, ft4], 
			axis = 0)

	else:
		Fm_corr = None
		Fm_corr_std = 0

	return (d13C_corr, 
			d13C_std_corr, 
			Fm_corr, 
			Fm_std_corr, 
			m_corr, 
			m_std_corr)

#define function to extract Rpo isotope data from .csv file
def _rpo_extract_iso(file, mass_err):
	'''
	Extracts mass, d13C, and Fm from a .csv file to be used for to create an
	``RpoIsotopes`` result instance.

	Parameters
	----------
	file : str or pd.DataFrame
		File containing isotope data, either as a path string or 
		``pd.DataFrame`` instance.

	mass_err : float
		Relative standard deviation on fraction masses. Defaults to 0.01 (i.e.
		1 percent of measured mass).

	Returns
	-------
	d13C : None or np.ndarray
		Array of d13C values for each fraction, length nFrac.
	
	d13C_std : np.ndarray
		Array of d13C stdev. for each fraction, length nFrac.

	Fm : None or np.ndarray
		Array of Fm values for each fraction, length nFrac.

	Fm_std : np.ndarray
		Array of Fm stdev. for each fraction, length nFrac.

	m : None or np.ndarray
		Array of masses (ugC) for each fraction, length nFrac.

	m_std : np.ndarray
		Array of mass stdev. (ugC) for each fraction, length nFrac.

	t : np.ndarray
		2d array of time for each fraction (in seconds), length nFrac.

	Raises
	------
	FileError
		If `file` does not contain "fraction" column.

	FileError
		If `file` is not str or ``pd.DataFrame``.
	
	FileError
		If index is not ``pd.DatetimeIndex`` instance.	
	
	FileError
		If first two rows are not fractions "-1" and "0"

	ScalarError
		If `mass_err` is not scalar.
	
	Notes
	-----
	For bookkeeping purposes, the first 2 rows must be fractions "-1" and "0",
	where the timestamp for fraction "-1" is the first point in `all_data` and
	the timestamp for fraction "0" is the t0 for the first fraction.
	'''

	#import file as a pd.DataFrame if inputted as a string path and check
	#that it is in the right format
	if isinstance(file, str):
		file = pd.DataFrame.from_csv(file)

	elif not isinstance(file, pd.DataFrame):
		raise FileError(
			'file must be pd.DataFrame or path string')

	if 'fraction' not in file.columns:
		raise FileError(
			'file must have "fraction" column')

	if not isinstance(file.index, pd.DatetimeIndex):
		raise FileError(
			'file index must be DatetimeIndex')

	if file.fraction[0] != -1 or file.fraction[1] != 0:
		raise FileError(
			'First two rows must be fractions "-1" and "0"')

	if not isinstance(mass_err, (str, float)):
		raise ScalarError(
			'mass_err must be string or float')
	else:
		#ensure float
		mass_err = float(mass_err)

	#extract time data
	secs = (file.index - file.index[0]).seconds
	t0 = secs[1:-1]
	tf = secs[2:]
	nF = len(t0)

	t = np.column_stack((t0, tf))

	#extract mass and isotope data, if they exist
	if 'ug_frac' in file.columns:
		m = file.ug_frac[2:].values
		m_std = m*mass_err

	else:
		m = None
		m_std = None

	if 'd13C' in file.columns:
		d13C = file.d13C[2:].values
		d13C_std = file.d13C_std[2:].values

	else:
		d13C = None
		d13C_std = None

	if 'Fm' in file.columns:
		Fm = file.Fm[2:].values
		Fm_std = file.Fm_std[2:].values

	else:
		Fm = None
		Fm_std = None

	return (d13C,
			d13C_std,
			Fm,
			Fm_std,
			m,
			m_std,
			t)

#define function to correct d13C for kinetic fractionation
def _rpo_kie_corr(
	result,
	d13C,
	d13C_std,
	model,
	ratedata,
	DE = 0.0018):

	'''
	Corrects d13C values for each RPO fraction for kinetic isotope effects.

	Parameters
	----------
	result : rp.Results
		Result instance containing the start/stop times to be used to calculate
		E_frac

	d13C : np.ndarray
		Array of d13C values to correct.

	d13C_std : np.ndarray
		Array of the standard deviation of d13C values to correct.

	model : rp.Model
		``rp.Model`` instance containing the A matrix to use for inversion.

	ratedata : rp.RateData
		``rp.Ratedata`` instance containing the reactive continuum data.

	DE : scalar
		Value for the difference in E between 12C- and 13C-containing
		atoms, in kJ. Defaults to 0.0018 (the best-fit value calculated
		in Hemingway et al., **2017**).

	Returns
	-------
	d13C_corr : np.ndarray
		Array of the fractionation-corrected d13C values (VPDB) of each 
		measured fraction, length `nFrac`.

	d13C_corr_std : np.ndarray
		The standard deviation of `d13C_corr` with length `nFrac`.
	'''

	#generate daem for 13C-containing atoms
	daem13 = Daem(model.E+DE, model.log10k0, model.t, model.T)

	#calculate ratio of rates, the KIE
	tg12 = np.dot(model.A,ratedata.p)
	tg13 = np.dot(daem13.A,ratedata.p)
	r = np.gradient(tg13)/np.gradient(tg12)

	#for each time slice, calculate KIE-corrected d13C
	d13C_corr = []

	#calculate cutoff indices
	ind_min, ind_max = _calc_cutoff(result, model)

	#loop through and correct each measurement
	for mi, ma, d13i in zip(ind_min, ind_max, d13C):
		
		#weighted-average kie for each slice
		kie_i = np.average(r[mi:ma], weights = -np.gradient(tg12)[mi:ma])

		#convert measured d13C to ratio
		R13_i = (d13i/1000 + 1)*0.011237

		#divide by kie_i to correct
		R13_i_corr = R13_i/kie_i

		#convert back to d13C
		d13C_i_corr = (R13_i_corr/0.011237 - 1)*1000

		#append list
		d13C_corr.append(d13C_i_corr)

	#since uncertainty is unknown, d13C_std is unchanged
	d13C_corr_std = d13C_std

	return d13C_corr, d13C_corr_std

#define function to correct d13C for isotope mass balance
def _rpo_mass_bal_corr(
	d13C,
	d13C_std,
	m,
	m_std,
	bulk_d13C_true):
	'''
	Corrects d13C values for the difference between mass-weighted mean and
	independently measured bulk values (i.e. isotope mass balance).

	Parameters
	----------
	d13C : np.ndarray
		Array of the d13C values (VPDB) of each measured fraction, 
		length `nFrac`.

	d13C_std : np.ndarray
		The standard deviation of `d13C` with length `nFrac`.

	m : np.ndarray
		Array of the masses (ugC) of each measured fraction, length `nFrac`.

	m_frac : np.ndarray
		The standard deviation of `m_frac` with length `nFrac`.

	bulk_d13C_true: array
		The true, independently measured bulk d13C value for the sample.
		Inputted in the form [mean, stdev.].

	Returns
	-------
	d13C_corr : np.ndarray
		Array of the mass-balance-corrected d13C values (VPDB) of each
		measured fraction, length `nFrac`.

	d13C_corr_std : np.ndarray
		The standard deviation of `d13C_corr` with length `nFrac`.

	Raises
	------
	ArrayError
		If `bulk_d13C_true` is not array-like

	LengthError
		If `bulk_d13C_true` is not an array of length 2.

	References
	----------
	[1] J.D. Hemingway et al. (2017) Assessing the blank carbon contribution, 
		isotope mass balance, and kinetic isotope fractionation of the ramped 
		pyrolysis/oxidation instrument at NOSAMS. **Radiocarbon**
	'''

	#ensure bulk_d13C_true is in the right format, and raise error if not
	if not isinstance(bulk_d13C_true, Sequence) and not hasattr(bulk_d13C_true, '__array__'):
		raise ArrayError(
			'bulk_d13C_true must be array-like')

	if len(bulk_d13C_true) != 2:
		raise LengthError(
			'length of bulk_d13C_true must be 2 [mean, stdev.]')

	#calculate the fractional contribution by each RPO fraction
	f = m/np.sum(m)

	#calculate the weighted-average d13C, d13C_std
	d13C_wgh = np.average(d13C, weights = f)
	d13C_wgh_std = norm(f*d13C_std)

	#calculate mass-balance-corrected values
	d13C_corr = d13C + (bulk_d13C_true[0] - d13C_wgh)
	d13C_corr_std = (d13C_std**2 + bulk_d13C_true[1]**2 + d13C_wgh_std**2)**0.5

	return d13C_corr, d13C_corr_std

