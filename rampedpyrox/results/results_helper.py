#* TODO: Test _kie_d13C_MC memory allocation and improve speed!

'''
This module contains helper functions for the Results class.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_d13C_to_R13', '_kie_d13C', '_kie_d13C_MC', '_nnls_MC', 
			'_R13_CO2', '_R13_diff', '_R13_to_d13C', '_rpo_blk_corr',
			'_rpo_cont_ctf', '_rpo_extract_iso']

import numpy as np
import pandas as pd
import warnings

from numpy.linalg import norm
from scipy.optimize import least_squares
from scipy.optimize import nnls

#import exceptions
from ..core.exceptions import(
	FileError,
	LengthError,
	ScalarError,
	)

#import helper functions
from ..core.core_functions import(
	assert_len
	)

from ..ratedata.ratedata_helper import(
	_calc_phi
	)

#define a function to convert d13C to 13C/12C ratio.
def _d13C_to_R13(d13C):
	'''
	Converts d13C values to 13R values using VPDB standard.

	Parameters
	----------
	d13C : np.ndarray
		Inputted d13C values, in per mille VPDB.

	Returns
	-------
	R13 : np.ndarray
		Corresponding 13C/12C ratios.
	'''

	#assert d13C is array with float dtype
	d13C = assert_len(d13C, len(d13C))


	Rpdb = 0.011237 #13C/12C ratio VPDB

	R13 = (d13C/1000 + 1)*Rpdb

	return R13

#define a function to calculate the d13C of each component, incorporating any KIE
def _kie_d13C(DEa, ind_wgh, model, ratedata, vals):
	'''
	Calculates the d13C of each component, accounting for any KIE fractionation.

	Parameters
	----------
	DEa : np.ndarray
		Array of DEa values (in kJ/mol) for each component in timedata.

	ind_wgh : np.ndarray
		Array of the mass-weighted center indices of each fraction.

	model : rp.Model
		``rp.Model`` instance containing the proper inversion model.
		Used to calculate 13C rates.

	ratedata : rp.RateData
		``rp.Ratedata`` instance containing the rate distribution leading
		to the KIE.

	vals : np.ndarray
		Array of fraction d13C values, length `nFrac`.

	Returns
	-------
	d13C_cmpt : np.ndarray
		Best-fit 13C/12C ratios for each component as determined by
		``scipy.optimize.least_squares`` and converted to d13C VPDB scale.

	d13C_err : float
		Fitting err determined as ``norm(Ax-b)``, and converted
		to d13C VPDB scale.

	Warnings
	--------
	UserWarning
		If ``scipy.optimize.least_squares`` cannot converge on a
		best-fit solution.
	'''

	#extract shapes -- NPEAKS AFTER COMBINED!
	_, nCmpt = np.shape(ratedata.peaks)

	#set initial guess of 0 permille
	r0 = _d13C_to_R13(np.zeros(nCmpt))

	#convert fraction d13C to R13
	R13_frac = _d13C_to_R13(vals)

	#perform fit
	res = least_squares(
		_R13_diff, 
		r0,
		bounds = (0, np.inf),
		args = (DEa, ind_wgh, model, R13_frac, ratedata))

	#ensure success
	if not res.success:
		warnings.warn(
			'R13 component calc. could not converge on a successful fit',
			UserWarning)

	#extract best-fit result
	R13_cmpt = res.x
	d13C_cmpt = _R13_to_d13C(R13_cmpt)

	#calculate predicted R13 of each fraction and convert to d13C
	R13_frac_pred = res.fun + R13_frac
	d13C_frac_pred = _R13_to_d13C(R13_frac_pred)

	#calculate err
	d13C_err = norm(vals - d13C_frac_pred)

	return d13C_cmpt, d13C_err

#define a function to run _kie_d13C in Monte Carlo fashion
def _kie_d13C_MC(DEa, ind_wgh, model, nIter, result, ratedata):
	'''
	Calculates the d13C of each component, accounting for any KIE fractionation,
	and bootstraps uncertainty.

	Parameters
	----------
	DEa : np.ndarray
		Array of DEa values (in kJ/mol) for each component in timedata.

	ind_wgh : np.ndarray
		Array of the mass-weighted center indices of each fraction.

	model : rp.Model
		``rp.Model`` instance containing the proper inversion model.
		Used to calculate 13C rates.

	nIter : int
		The number of times to iterate.

	result : rp.Result
		``rp.Result`` instance containing the fraction isotopes used for
		deconvolution.

	ratedata : rp.RateData
		``rp.Ratedata`` instance containing the rate distribution leading
		to the KIE.

	Returns
	-------
	pk_val : np.ndarray
		Resulting estimated component isotope values, length `nCmpt`.

	pk_std : np.ndarray
		Resulting estimated component isotope stdev., length `nCmpt`.

	rmse : float
		Average RMSE between the measured and predicted fraction isotopes.
	
	Raises
	------
	LengthError
		If length of `ind_wgh` is not `nFrac`.
	'''

	################################################################
	# assert all datatypes here as to prevent checking nIter times #
	################################################################
	
	#assert DEa and ind_wgh are the right length
	DEa = assert_len(DEa, ratedata._pkinf.shape[0])
	
	if len(ind_wgh) != result.nFrac:
		raise LengthError(
			'ind_wgh of length = %r is not compatable with a model with'
			' nFrac = %r. Ensure this is the right model instance'
			%(len(ind_wgh), result.nFrac))

	#ensure that ind_wgh and nIter dtype is integer
	ind_wgh = [int(i) for i in ind_wgh]
	nIter = int(nIter)

	#nPeaks AFTER BEING COMBINED!
	_, nCmpt = np.shape(ratedata.peaks)
	nFrac = result.nFrac

	#extract data
	vals =  result.d13C_frac.reshape(nFrac, 1)
	vals_std = result.d13C_frac_std.reshape(nFrac, 1)

	#generate noise matrix
	noise = np.random.standard_normal(size = (nFrac, nIter))
	
	#generate noisy fraction isotoes
	vals_MC = vals + vals_std*noise

	#pre-allocate results
	pks = np.zeros([nIter, nCmpt])
	errs = np.zeros(nIter)

	#loop through and store each iteration
	for i, v in enumerate(vals_MC.T):

		#calculate result
		res = _kie_d13C(DEa, ind_wgh, model, ratedata, v)

		#store result
		pks[i] = res[0]
		errs[i] = res[1]

	#calculate statistics
	pk_val = np.mean(pks, axis = 0)
	pk_std = np.std(pks, axis = 0)

	rmse = np.mean(errs)/(nFrac**0.5) 

	return pk_val, pk_std, rmse

#define a function to run nnls in Monte Carlo fashion
def _nnls_MC(cont, nIter, vals, vals_std):
	'''
	Calculates the component mass or Fm using nnls and Monte Carlo.
	
	Parameters
	----------
	cont : np.ndarray
		2d array of the contribution of each component to each fraction (for Fm) or
		of each fraction to each component (for mass). Shape [`nFrac` x `nCmpt`].

	nIter : int
		The number of times to iterate.

	vals : np.ndarray
		Array of the isotope/mass values, length `nFrac`.

	vals_std :
		Array of the isotope/mass standard deviations, length `nFrac`.

	Returns
	-------
	pk_val : np.ndarray
		Resulting estimated component isotope/mass values, length `nCmpt`.

	pk_std : np.ndarray
		Resulting estimated component isotope/mass stdev., length `nCmpt`.

	rmse : float
		Average RMSE between the measured and predicted fraction isotopes/
		masses.
	'''

	#extract shapes and ensure lengths
	nFrac, nCmpt = cont.shape
	vals = assert_len(vals, nFrac)
	vals_std = assert_len(vals_std, nFrac)

	#generate noise matrix
	noise = np.random.standard_normal(size = (nFrac, nIter))
	
	#generate noisy fraction isotopes
	vals = vals.reshape(nFrac, 1)
	vals_std = vals_std.reshape(nFrac, 1)

	vals_MC = vals + vals_std*noise

	#pre-allocate results
	pks = np.zeros([nIter, nCmpt])
	errs = np.zeros(nIter)

	#loop through and store each iteration
	for i, v in enumerate(vals_MC.T):

		#calculate result
		res = nnls(cont, v)

		#store result
		pks[i] = res[0]
		errs[i] = res[1]

	#calculate statistics
	pk_val = np.mean(pks, axis = 0)
	pk_std = np.std(pks, axis = 0)

	rmse = np.mean(errs)/(nFrac**0.5) 

	return pk_val, pk_std, rmse

#define a function to calculate CO2 13C/12C ratios.
def _R13_CO2(DEa, model, R13_cmpt, ratedata):
	'''
	Calculates the 13C/12C ratio for instantaneously eluted CO2 at each
	timepoint for a given 13C/12C ratio of each component.
	
	Parameters
	----------
	DEa : np.ndarray
		Array of DEa values (in kJ/mol) for each peak, length nPeak.

	model : rp.Model
		``rp.Model`` instance containing the model to generate forward-
		modeled 12C and 13C decomposition rates.

	R13_cmpt : np.ndarray
		13C/12C ratio for each component, length `nCmpt`.

	ratedata : rp.RateData
		``rp.RateData`` instance containing the k/Ea distribution to use for
		calculating the KIE.

	Returns
	-------
	R13_CO2 : np.ndarray
		Array of 13C/12C ratio of instantaneously eluted CO2 at each 
		timepoint, length `nt`.
	'''

	#extract k/Ea (necessary since models have different nomenclature)
	if hasattr(ratedata, 'k'):
		k = ratedata.k
	elif hasattr(ratedata, 'Ea'):
		k = ratedata.Ea
	
	#extract 12C and 13C peaks from ratedata
	C12_mu = ratedata._pkinf[:,0]
	sigma = ratedata._pkinf[:,1]
	C12_height = ratedata._pkinf[:,2]

	#if peaks have been combined, repeat R13_peak as necessary
	if ratedata._cmbd is not None:

		#calculate indices of deleted peaks
		dp = [val - i for i, val in enumerate(ratedata._cmbd)]
		dp = np.array(dp) #convert to nparray
		
		#insert deleted peaks back in
		R13_cmpt = np.insert(R13_cmpt, dp, R13_cmpt[dp-1])

	#calculate C13 means and heights
	C13_mu = C12_mu + DEa
	C13_height = C12_height*R13_cmpt

	#calculate the rate distribution for C12 and C13
	C12_phi, _ = _calc_phi(
		k, 
		C12_mu, 
		sigma, 
		C12_height, 
		ratedata.peak_shape)

	C13_phi, _ = _calc_phi(
		k, 
		C13_mu, 
		sigma, 
		C13_height, 
		ratedata.peak_shape)

	#forward-model 13C and 12C gam
	C12_gam = np.inner(model.A, C12_phi)
	C13_gam = np.inner(model.A, C13_phi)

	#convert to 13C and 12C thermograms, and calculate R13_CO2
	C12_dgamdt = -np.gradient(C12_gam)
	C13_dgamdt = -np.gradient(C13_gam)

	return C13_dgamdt/C12_dgamdt

#define a function to calculate true - predicted R13 difference
def _R13_diff(R13_cmpt, DEa, ind_wgh, model, R13_frac, ratedata):
	'''
	Calculates the difference between measured and predicted 13C/12C ratio. 
	To be used by ``scipy.optimize.least_squares``.

	Parameters
	----------
	R13_cmpt : np.ndarray
		13C/12C ratio for each component, length nCmpt.

	DEa : np.ndarray
		Array of DEa values (in kJ/mol) for each peak.

	ind_wgh : np.ndarray
		Index in ``timedata.t`` corresponding to the mass-weighted mean time
		for each fraction. Length nFrac.

	model : rp.Model
		``rp.Model`` instance containing the proper inversion model.
		Used to calculate 13C rates.

	R13_frac : np.ndarray
		Array of 13C/12C ratios for each fraction, length nFrac.

	ratedata : rp.RateData
		``rp.Ratedata`` instance containing the rate distribution leading
		to the KIE.

	Returns
	-------
	R13_diff : np.ndarray
		Difference between measured and predicted 13C/12C ratio for each 
		fraction, length nFrac.
	'''

	#calculate the R13 of instantaneously produced CO2 at each point
	R13_CO2 = _R13_CO2(
		DEa, 
		model, 
		R13_cmpt, 
		ratedata)

	#calculate difference at ind_wgh
	R13_diff = R13_CO2[ind_wgh] - R13_frac

	return R13_diff

#define a function to convert 13C/12C ratio to d13C.
def _R13_to_d13C(R13):
	'''
	Converts 13R values to d13C values using VPDB standard.

	Parameters
	----------
	R13 : np.ndarray
		13C/12C ratio values to be converted to d13C in VPDB scale.

	Returns
	-------
	d13C : np.ndarray
		Resulting d13C values.
	'''

	#assert R13 is array with float dtype
	R13 = assert_len(R13, len(R13))

	Rpdb = 0.011237 #13C/12C ratio VPDB

	d13C = (R13/Rpdb - 1)*1000

	return d13C

#define a function to blank-correct fraction isotopes
def _rpo_blk_corr(d13C, d13C_std, Fm, Fm_std, m, m_std, t):
	'''
	Performs blank correction (NOSAMS RPO instrument) on raw isotope values.

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
	[1] J.D. Hemingway et al. **(in prep)** Assessing the blank carbon
		contribution, isotope mass balance, and kinetic isotope fractionation
		of the ramped pyrolysis/oxidation instrument at NOSAMS.
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

#define function to calculate the component to fraction contribution.
def _rpo_cont_ctf(result, timedata, ctf = True):
	'''
	Calculates the contribution of each component to each fraction or of each
	fraction to each component.

	Parameters
	----------
	result : rp.RpoIsotopes
		``RpoIsotopes`` instance containing CO2 fraction information.

	timedata : rp.TimeData
		``TimeData`` instance containing the thermogram of interest.

	ctf : Boolean
		If True, calculates the contribution of each component to each fraction.
		If False, calculates the contribution of each fraction to each component.

	Returns
	-------
	cont_ctf : np.ndarray
		Array of the contribution by each component to each measured CO2
		fraction or each fraction to each component, shape [`nFrac` x `nCmpt`].

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
	UserWarning
		If nCmpt is greater than nFrac, the problem is underconstrained.

	Notes
	-----
	This method uses peaks **after** the "combined"  flag has been 
	implemented. That is, it treats combined peaks as a single  component when 
	calculating indices and contributions to each fraction.
	'''

	#extract shapes
	nFrac = result.nFrac
	nCmpt = timedata.nCmpt
	nt = timedata.nt
	
	#extract arrays
	t_frac = result.t_frac
	t = timedata.t
	dt = np.gradient(t).reshape(nt,1)
	wgh = -timedata.dgamdt
	cmpts = -timedata.dcmptdt

	#raise warnings
	if nCmpt > nFrac:
		warnings.warn(
			'Warning: nCmpt = %r, nFrac = %r. Problem is underconstrained!'
			' Solution is not unique!' %(nCmpt, nFrac),
			UserWarning)

	#pre-allocate cont_ctf matrix and index arrays
	cont_ctf = np.zeros([nFrac,nCmpt])
	ind_min = np.zeros(nFrac, dtype = int)
	ind_max = np.zeros(nFrac, dtype = int)
	ind_wgh = np.zeros(nFrac, dtype = int)

	#loop through and calculate contributions and indices
	for i, row in enumerate(t_frac):

		#extract indices for each fraction
		ind = np.where((t > row[0]) & (t <= row[1]))[0]

		#store first and last indices
		ind_min[i] = ind[0]
		ind_max[i] = ind[-1]

		#calculate mass-weighted average index
		av = np.average(ind, weights = wgh[ind])
		ind_wgh[i] = int(np.round(av))

		if ctf is True:
			#calculate component to fraction contribution
			ctf_i = np.sum(cmpts[ind], axis = 0)/np.sum(wgh[ind])

		else:
			#calculate contribution of each fraction to each component
			ctf_i = np.sum(cmpts[ind]*dt[ind], axis = 0)/np.sum(cmpts*dt, 
				axis = 0)

		#store in row i
		cont_ctf[i] = ctf_i

	return cont_ctf, ind_min, ind_max, ind_wgh

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
		m_std = 0

	if 'd13C' in file.columns:
		d13C = file.d13C[2:].values
		d13C_std = file.d13C_std[2:].values

	else:
		d13C = None
		m_std = 0

	if 'Fm' in file.columns:
		Fm = file.Fm[2:].values
		Fm_std = file.Fm_std[2:].values

	else:
		Fm = None
		Fm_std = 0

	return (d13C, 
			d13C_std, 
			Fm, 
			Fm_std, 
			m, 
			m_std, 
			t)
