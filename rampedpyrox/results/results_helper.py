'''
This module contains helper functions for the Results class.
'''

import numpy as np
import pandas as pd
import warnings

from numpy.linalg import norm
from scipy.optimize import least_squares

#define a function to convert d13C to 13C/12C ratio.
def _d13C_to_R13(d13C):
	'''
	Converts d13C values to 13R values using VPDB standard.
	Called by ``blank_correct()``.
	Called by ``_extract_isotopes()``.

	Parameters
	----------
	d13C : np.ndarray
		Inputted d13C values, in per mille VPDB.

	Returns
	-------
	R13 : np.ndarray
		Corresponding 13C/12C ratios.
	'''

	Rpdb = 0.011237 #13C/12C ratio VPDB

	R13 = (d13C/1000 + 1)*Rpdb

	return R13

#define a function to calculate the d13C of each peak, incorporating any KIE
def _kie_d13C(DEa, ind_wgh, model, result, ratedata):
	'''
	Calculates the d13C of each peak, accounting for any KIE fractionation.

	Parameters
	----------
	DEa : np.ndarray
		Array of DEa values (in kJ/mol) for each peak in timedata.

	ind_wgh : np.ndarray
		Array of the mass-weighted center indices of each fraction.

	result : rp.Result
		Result instance containing d13C fraction data of interest.

	timedata : rp.TimeData
		TimeData instance containing timedata of interest.

	Returns
	-------
	d13C_peak : np.ndarray
		Best-fit peak 13C/12C ratios for each peak as determined by
		``scipy.optimize.least_squares()`` and converted to d13C VPDB scale.

	d13C_err : float
		Fitting err determined as ``norm(Ax-b)``, and converted
		to d13C VPDB scale.

	Warnings
	--------
	Raises warning if ``scipy.optimize.least_squares`` cannot converge on a
	best-fit solution.

	References
	----------
	'''

	#extract shapes
	_, nPeak = np.shape(ratedata.peaks)

	#set initial guess of 0 permille
	r0 = _d13C_to_R13(np.ones(nPeak))

	#convert fraction d13C to R13
	R13_frac = _d13C_to_R13(result.d13C_frac)

	#perform fit
	res = least_squares(_R13_diff, r0,
		bounds = (0, np.inf),
		args = (DEa, ind_wgh, model, R13_frac, ratedata))

	#ensure success
	if not res.success:
		warnings.warn('R13 peak calc. could not converge on a successful fit')

	#extract best-fit result
	R13_peak = res.x
	d13C_peak = _R13_to_d13C(R13_peak)

	#calculate predicted R13 of each fraction and convert to d13C
	R13_frac_pred = res.fun + R13_frac
	d13C_frac_pred = _R13_to_d13C(R13_frac_pred)

	#calculate err
	d13C_err = norm(result.d13C_frac - d13C_frac_pred)

	return d13C_peak, d13C_err

#define a function to calculate CO2 13C/12C ratios.
def _R13_CO2(DEa, model, R13_peak, ratedata):
	'''
	Calculates the 13C/12C ratio for instantaneously eluted CO2 at each
	timepoint for a given 13C/12C ratio of each peak.
	
	Parameters
	----------
	DEa : np.ndarray
		Array of DEa values (in kJ/mol) for each peak, length nPeak.

	model : rp.Model


	R13_peak : np.ndarray
		13C/12C ratio for each peak, length nPeak.

	ratedata : rp.RateData
		RateData instance containing the k/Ea distribution to use for
		calculating the KIE.

	Returns
	-------
	R13_CO2 : np.ndarray
		Array of 13C/12C ratio of instantaneously eluted CO2 at each 
		timepoint, length nt.

	Raises
	------
	ValueError
		If `R13C_peak` is not of length nPeak (after `combined`).
	'''

	#extract shapes and assert type/shape
	_, nPeak = np.shape(ratedata.peaks)

	if not isinstance(R13_peak, np.ndarray) or len(R13_peak) != nPeak:
		raise ValueError('R13_peak must be array with len. nPeak (combined)')
	
	#extract 12C and 13C peaks and scale to correct heights
	C12_peaks_scl = ratedata.peaks
	C13_peaks_scl = (ratedata.peaks + DEa)*R13_peak

	#sum to create scaled phi arrays
	C12_phi_scl = np.sum(C12_peaks_scl, axis = 1)
	C13_phi_scl = np.sum(C13_peaks_scl, axis = 1)

	#forward-model 13C and 12C gam
	C12_gam = np.inner(model.A, C12_phi_scl)
	C13_gam = np.inner(model.A, C13_phi_scl)

	#convert to 13C and 12C thermograms, and calculate R13_CO2
	C12_dgamdt = -np.gradient(C12_gam)
	C13_dgamdt = -np.gradient(C13_gam)

	return C13_dgamdt/C12_dgamdt

#define a function to calculate true - predicted R13 difference
def _R13_diff(R13_peak, DEa, ind_wgh, model, R13_frac, ratedata):
	'''
	Calculates the difference between measured and predicted 13C/12C ratio. 
	To be used by ``scipy.optimize.least_squares``.

	Parameters
	----------
	R13_peak : np.ndarray
		13C/12C ratio for each peak, length nPeaks.

	R13_frac : np.ndarray
		Array of 13C/12C ratios for each fraction, length nFrac.

	ind_wgh : np.ndarray
		Index in ``timedata.t`` corresponding to the mass-weighted mean time
		for each fraction. Length nFrac.

	timedata : rp.TimeData
		TimeData instance containing the timedata of interest.

	Returns
	-------
	R13_diff : np.ndarray
		Difference between measured and predicted 13C/12C ratio for each 
		fraction, length nFrac.
	'''

	R13_CO2 = _R13_CO2(DEa, model, R13_peak, ratedata)

	R13_diff = R13_CO2[ind_wgh] - R13_frac

	return R13_diff

#define a function to convert 13C/12C ratio to d13C.
def _R13_to_d13C(R13):
	'''
	Converts 13R values to d13C values using VPDB standard.
	Called by ``IsotopeResult.__init__()``.
	Called by ``_fit_R13_peak()``.

	Parameters
	----------
	R13 : np.ndarray
		13C/12C ratio values to be converted to d13C in VPDB scale.

	Returns
	-------
	d13C : np.ndarray
		Resulting d13C values.
	'''

	Rpdb = 0.011237 #13C/12C ratio VPDB

	d13C = (R13/Rpdb - 1)*1000

	return d13C

#define a function to blank-correct fraction isotopes
def _rpo_blk_corr(d13C, d13C_std, Fm, Fm_std, m, m_std, t):
	'''
	Performs blank correction (NOSAMS RPO instrument) on raw isotope values.

	Parameters
	----------
	d13C : np.ndarray
		Array of d13C values for each fraction, length nFrac.
	
	d13C_std : np.ndarray
		Array of d13C stdev. for each fraction, length nFrac.

	Fm : np.ndarray
		Array of Fm values for each fraction, length nFrac.

	Fm_std : np.ndarray
		Array of Fm stdev. for each fraction, length nFrac.

	m : np.ndarray
		Array of masses (ugC) for each fraction, length nFrac.

	m_std : np.ndarray
		Array of mass stdev. (ugC) for each fraction, length nFrac.

	t : np.ndarray
		2d array of time for each fraction (in seconds), length nFrac.

	Returns
	-------
	d13C_corr : np.ndarray
		Array of corrected d13C values for each fraction, length nFrac.
	
	d13C_std_corr : np.ndarray
		Array of corrected d13C stdev. for each fraction, length nFrac.

	Fm_corr : np.ndarray
		Array of corrected Fm values for each fraction, length nFrac.

	Fm_std_corr : np.ndarray
		Array of corrected Fm stdev. for each fraction, length nFrac.

	m_corr : np.ndarray
		Array of corrected masses (ugC) for each fraction, length nFrac.

	m_std_corr : np.ndarray
		Array of corrected mass stdev. (ugC) for each fraction, length nFrac.
	
	References
	----------
	J.D. Hemingway et al. **(in prep)** Assessing the blank carbon
	contribution, isotope mass balance, and kinetic isotope fractionation of 
	the ramped pyrolysis/oxidation instrument at NOSAMS.
	'''

	#define constants
	bl_flux = 0.375/1000 #ug/s
	bl_flux_std = 5.83e-5

	bl_d13C = -29.0
	bl_d13C_std = 0.1

	bl_Fm = 0.555
	bl_Fm_std = 0.042

	#calculate blank mass for each fraction
	dt = t[:,1] - t[:,0]
	bl_mass = bl_flux*dt #ug

	#perform blank correction

	#correct mass
	m_corr = m - bl_mass
	m_std_corr = norm([m_std, dt*bl_flux_std], axis = 0)

	#correct d13C
	dt1 = d13C_std
	dt2 = dt*bl_d13C*bl_flux_std/m
	dt3 = bl_mass*bl_d13C_std/m
	dt4 = bl_mass*bl_d13C*m_std_corr/(m_corr**2)
	
	d13C_corr = (m*d13C - bl_mass*bl_d13C)/m_corr
	d13C_std_corr = norm([dt1, dt2, dt3, dt4], axis = 0)

	#correct Fm
	ft1 = Fm_std
	ft2 = dt*bl_Fm*bl_flux_std/m
	ft3 = bl_mass*bl_Fm_std/m
	ft4 = bl_mass*bl_Fm*m_std_corr/(m_corr**2)
	
	Fm_corr = (m*Fm - bl_mass*bl_Fm)/m_corr
	Fm_std_corr = norm([ft1, ft2, ft3, ft4], axis = 0)

	return d13C_corr, d13C_std_corr, Fm_corr, Fm_std_corr, m_corr, m_std_corr

#define function to peak to fraction contribution.
def _rpo_cont_ptf(result, timedata):
	'''
	Calculates the contribution of each peak to each fraction.

	Parameters
	----------
	result : rp.RpoIsotopes
		``RpoIsotopes`` instance containing CO2 fraction information.

	timedata : rp.TimeData
		``TimeData`` instance containing the thermogram of interest.

	Returns
	-------
	cont_ptf : np.ndarray
		Array of the contribution by each Ea peak to each measured CO2
		fraction, shape [nFrac x nPeak].

	ind_min : np.ndarray
		Index in ``timedata.t`` corresponding to the minimum time for each 
		fraction. Length nFrac.

	ind_max : np.ndarray
		Index in ``timedata.t`` corresponding to the maximum time for each 
		fraction. Length nFrac.

	ind_wgh : np.ndarray
		Index in ``timedata.t`` corresponding to the mass-weighted mean time
		for each fraction. Length nFrac.

	Warnings
	------
	Warns if nPeak is greater than nFrac, the problem is underconstrained.

	Notes
	-----
	This method uses peaks **after** the "combined"  flag has been 
	implemented. That is, it treats combined peaks as a single  peak when 
	calculating indices and contributions to each fraction.
	'''

	#extract shapes
	nFrac = result.nFrac
	nPeak = timedata.nPeak
	nt = timedata.nt
	
	#extract arrays
	t_frac = result.t_frac
	t = timedata.t
	wgh = -timedata.dgamdt
	peaks = -timedata.dcmptdt

	#raise errors
	if nPeak > nFrac:
		warnings.warn((
			"Warning: nPeak = %r, nFrac = %r. Problem is underconstrained!"
			"Solution is not unique!" %(nPeak, nFrac)))

	#pre-allocate cont_ptf matrix and index arrays
	cont_ptf = np.zeros([nFrac,nPeak])
	ind_min = []
	ind_max = []
	ind_wgh = []

	#loop through and calculate contributions and indices
	for i, row in enumerate(t_frac):

		#extract indices for each fraction
		ind = np.where((t > row[0]) & (t <= row[1]))[0]

		#store first and last indices
		ind_min.append(ind[0])
		ind_max.append(ind[-1])

		#calculate mass-weighted average index
		av = np.average(ind, weights = wgh[ind])
		ind_wgh.append(int(np.round(av)))

		#calculate peak to fraction contribution
		ptf_i = np.sum(peaks[ind], axis = 0)/np.sum(wgh[ind])

		#store in row i
		cont_ptf[i] = ptf_i

	return cont_ptf, ind_min, ind_max, ind_wgh

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
	d13C : np.ndarray
		Array of d13C values for each fraction, length nFrac.
	
	d13C_std : np.ndarray
		Array of d13C stdev. for each fraction, length nFrac.

	Fm : np.ndarray
		Array of Fm values for each fraction, length nFrac.

	Fm_std : np.ndarray
		Array of Fm stdev. for each fraction, length nFrac.

	m : np.ndarray
		Array of masses (ugC) for each fraction, length nFrac.

	m_std : np.ndarray
		Array of mass stdev. (ugC) for each fraction, length nFrac.

	t : np.ndarray
		2d array of time for each fraction (in seconds), length nFrac.

	Raises
	------
	TypeError
		If `file` is not str or ``pd.DataFrame``.
	
	TypeError
		If index is not ``pd.DatetimeIndex`` instance.	

	TypeError
		If `mass_err` is not scalar.

	ValueError
		If `file` does not contain "d13C", "d13C_std", "Fm", "Fm_std", 
		"ug_frac", and "fraction" columns.
	
	ValueError
		If first two rows are not fractions "-1" and "0"
	
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
		raise TypeError('file must be pd.DataFrame or path string')

	if 'fraction' and 'd13C' and 'd13C_std' and 'Fm' and 'Fm_std' and \
		'ug_frac' not in file.columns:
		raise ValueError((
			"file must have 'fraction', 'd13C', 'd13C_std', 'Fm', 'Fm_std'," 
			"and 'ug_frac' columns"))

	if not isinstance(file.index,pd.DatetimeIndex):
		raise TypeError('file index must be DatetimeIndex')

	if file.fraction[0] != -1 or file.fraction[1] != 0:
		raise ValueError('First two rows must be fractions "-1" and "0"')

	if not isinstance(mass_err, (str, float)):
		raise TypeError('mass_err must be string or float')
	else:
		#ensure float
		mass_err = float(mass_err)

	#extract time data
	secs = (file.index - file.index[0]).seconds
	t0 = secs[1:-1]
	tf = secs[2:]
	nF = len(t0)

	t = np.column_stack((t0, tf))

	#extract mass and isotope data
	m = file.ug_frac[2:].values
	m_std = m*mass_err

	d13C = file.d13C[2:].values
	d13C_std = file.d13C_std[2:].values

	Fm = file.Fm[2:].values
	Fm_std = file.Fm_std[2:].values

	return d13C, d13C_std, Fm, Fm_std, m, m_std, t







