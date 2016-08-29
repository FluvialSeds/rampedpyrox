'''
``isotoperesult`` module for calculating the isotope composition of individual
Ea peaks within a sample. Stores data in a ``IsotopeResult`` instance.
'''

from __future__ import print_function

import numpy as np
import pandas as pd
import warnings

from numpy.linalg import norm
from scipy.optimize import least_squares
from scipy.optimize import nnls

__docformat__ = 'restructuredtext en'

## PRIVATE FUNCTIONS ##

#define function to peak to fraction contribution.
def _calc_cont_ptf(lt, ec, t0_frac, tf_frac):
	'''
	Calculates the contribution of each peak to each fraction.
	Called by ``IsotopeResult.__init__()``.

	Parameters
	----------
	lt : rp.LapalceTransform)
		``rp.LaplaceTransform`` instance containing the Laplace transform 
		matrix to convert f(Ea) peaks to carbon degradation rates at each 
		timepoint.

	ec : rp.EnergyComplex)
		``rp.EnergyComplex`` instance containing f(Ea) Gaussian peaks.

	t0_frac : np.ndarray
		Array of t0 for each fraction (in seconds), length nFrac.

	tf_frac : np.ndarray
		Array of tf for each fraction (in seconds), length nFrac.

	Returns
	-------
	cont_ptf : np.ndarray)
		Array of the contribution by each Gaussian Ea peak to each measured
		CO2 fraction, shape [nFrac x nPeak].

	ind_min : np.ndarray
		Index in ``lt.t`` corresponding to the minimum time for each fraction.
		Length nFrac.

	ind_max : np.ndarray
		Index in ``lt.t`` corresponding to the maximum time for each fraction.
		Length nFrac.

	ind_wgh : np.ndarray
		Index in ``lt.t`` corresponding to the mass-weighted mean time for
		each fraction. Length nFrac.

	Raises
	------
	ValueError
		If nPeaks is greater than nFrac, the problem is underconstrained.

	Notes
	-----
	This method uses ``rp.EnergyComplex`` peaks **after** the "comnine_last" 
	flag has been implemented. That is, it treats combined peaks as a single 
	peak when calculating indices and contributions to each fraction.
	'''

	#extract shapes
	nT,nE = np.shape(lt.A)
	nFrac = len(t0_frac)
	_,nPeak = np.shape(ec.peaks) #AFTER PEAKS HAVE BEEN COMBINED!

	#combine t0 and tf
	t = np.column_stack((t0_frac,tf_frac))

	#raise errors
	if nPeak > nFrac:
		raise ValueError('Under constrained problem! nPeaks > nFractions!!')

	#calculate modeled g using lt and ec
	t_tg = lt.t
	g = np.inner(lt.A,ec.phi_hat) #fraction
	g_peak = np.inner(lt.A,ec.peaks.T) #fraction (each peak)

	#take the gradient to calculate the thermograms (per timestep!)
	tot = -np.gradient(g) #fraction/timestep
	
	#peaks = -np.gradient(g_peak, axis=0) #fraction/timestep (each peak)
	#not python2 compatible. Use the following instead:
	peaks = -np.gradient(g_peak)[0] #fraction/timestep (each peak)


	#pre-allocate cont_ptf matrix and index arrays
	cont_ptf = np.zeros([nFrac,nPeak])
	ind_min = []
	ind_max = []
	ind_wgh = []

	#loop through and calculate contributions and indices
	for i,row in enumerate(t):

		#extract indices for each fraction
		ind = np.where((t_tg > row[0]) & (t_tg <= row[1]))[0]

		#store first and last indices
		ind_min.append(ind[0])
		ind_max.append(ind[-1])

		#calculate mass-weighted average index
		av = np.average(ind, weights=tot[ind])
		ind_wgh.append(int(np.round(av)))

		#calculate peak to fraction contribution
		ptf_i = np.sum(peaks[ind],axis=0)/np.sum(tot[ind])

		#store in row i
		cont_ptf[i] = ptf_i

	return cont_ptf, ind_min, ind_max, ind_wgh

#define a function to calculate CO2 13C/12C ratios.
def _calc_R13_CO2(R13_peak, lt, ec):
	'''
	Calculates the 13C/12C ratio for instantaneously eluted CO2 at each
	timepoint for a given 13C/12C ratio of each peak.
	Called by ``_R13_diff()``.
	
	Parameters
	----------
	R13_peak : np.ndarray
		13C/12C ratio for each peak, length nPeaks.

	lt : rp.LapalceTransform)
		``rp.LaplaceTransform`` instance containing the Laplace transform 
		matrix to convert f(Ea) peaks to carbon degradation rates at each 
		timepoint.

	ec : rp.EnergyComplex)
		``rp.EnergyComplex`` instance containing f(Ea) Gaussian peaks.

	Returns
	-------
	R13_CO2 : np.ndarray
		Array of 13C/12C ratio of instantaneously eluted CO2 at each 
		timepoint, length nT.

	Raises
	------
	ValueError
		If `R13C_peak` is not of length nPeak (after `combine_last`).
	'''

	#check R13_peak
	_,nPeak = np.shape(ec.peaks)

	if not isinstance(R13_peak, np.ndarray) or len(R13_peak) != nPeak:
		raise ValueError('R13_peak must be array with len. nPeak (combined)')

	eps = ec.eps
	
	#extract 12C and 13C Ea Gaussian peaks and scale to correct heights
	C12_peaks_scl = ec.peaks
	C13_peaks_scl = ec._peaks_13*R13_peak

	#sum to create scaled phi_hat arrays
	phi_hat_12_scl = np.sum(C12_peaks_scl,axis=1)
	phi_hat_13_scl = np.sum(C13_peaks_scl,axis=1)

	#forward-model 13C and 12C g_hat
	g_hat_12 = np.inner(lt.A,phi_hat_12_scl)
	g_hat_13 = np.inner(lt.A,phi_hat_13_scl)

	#convert to 13C and 12C thermograms, and calculate R13_CO2
	grad_t = np.gradient(lt.t)
	gdot_hat_12 = -np.gradient(g_hat_12)/grad_t
	gdot_hat_13 = -np.gradient(g_hat_13)/grad_t

	R13_CO2 = gdot_hat_13/gdot_hat_12

	return R13_CO2

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

#define a function to extract isotopes and masses from sum_data.
def _extract_isotopes(sum_data, mass_rsd=0, add_noise=False):
	'''
	Extracts isotope data from the "sum_data" file.
	Called by ``IsotopeResult.__init__()``.

	Parameters
	----------
	sum_data : str or pd.DataFrame
		File containing isotope data, either as a path string or 
		``pd.DataFrame`` instance.

	mass_rsd : float
		Relative standard deviation on fraction masses. Defaults to 0.01 (i.e.
		1 percent of measured mass).

	add_noise : boolean
		Tells the program whether or not to add Gaussian noise to isotope and
		mass values. To be used for Monte Carlo uncertainty calculations. 
		Defaults to `False`.

	Returns
	-------
	t0_frac : np.ndarray
		Array of t0 for each fraction (in seconds), length nFrac.

	tf_frac : np.ndarray
		Array of tf for each fraction (in seconds), length nFrac.

	mass_frac : np.ndarray
		Array of masses (ugC) for each fraction, length nFrac.
	
	R13_frac : np.ndarray
		Array of 13C/12C ratios for each fraction, length nFrac.
	
	Fm_frac : np.ndarray
		Array of Fm values for each fraction, length nFrac.


	Raises
	------
	ValueError
		If `sum_data` is not str or ``pd.DataFrame``.
	
	ValueError
		If `sum_data` does not contain "d13C", "d13C_std", "Fm", "Fm_std", 
		"ug_frac", and "fraction" columns.
	
	ValueError
		If index is not ``pd.DatetimeIndex`` instance.
	
	ValueError
		If first two rows are not fractions "-1" and "0"

	Notes
	-----
	For bookkeeping purposes, the first 2 rows must be fractions "-1" and "0",
	where the timestamp for fraction "-1" is the first point in `all_data` and
	the timestamp for fraction "0" is the t0 for the first fraction.
	'''

	#import sum_data as a pd.DataFrame if inputted as a string path and check
	#that it is in the right format
	if isinstance(sum_data,str):
		sum_data = pd.DataFrame.from_csv(sum_data)

	elif not isinstance(sum_data,pd.DataFrame):
		raise ValueError('sum_data must be pd.DataFrame or path string')

	if 'fraction' and 'd13C' and 'd13C_std' and 'Fm' and 'Fm_std' and \
		'ug_frac' not in sum_data.columns:
		raise ValueError('sum_data must have "fraction", "d13C", "d13C_std",'\
			' "Fm", "Fm_std", and "ug_frac" columns')

	if not isinstance(sum_data.index,pd.DatetimeIndex):
		raise ValueError('sum_data index must be DatetimeIndex')

	if sum_data.fraction[0] != -1 or sum_data.fraction[1] != 0:
		raise ValueError('First two rows must be fractions "-1" and "0"')

	#extract time data
	secs = (sum_data.index - sum_data.index[0]).seconds
	t0_frac = secs[1:-1]
	tf_frac = secs[2:]
	nF = len(t0_frac)

	#extract mass and isotope data
	mass_frac = sum_data.ug_frac[2:].values
	d13C_frac = sum_data.d13C[2:].values
	Fm_frac = sum_data.Fm[2:].values

	#extract standard deviations
	if add_noise:
		mass_frac_std = mass_frac*mass_rsd
		d13C_frac_std = sum_data.d13C_std[2:].values
		Fm_frac_std = sum_data.Fm_std[2:].values
		sigs = np.column_stack((mass_frac_std,d13C_frac_std,Fm_frac_std))
	else:
		sigs = np.zeros([nF,3])

	#generate noise and add to data
	np.random.seed()
	err = np.random.randn(nF,3)*sigs
	mass_frac = mass_frac + err[:,0]
	d13C_frac = d13C_frac + err[:,1]
	Fm_frac = Fm_frac + err[:,2]

	#convert d13C to 13C/12C ratio
	R13_frac = _d13C_to_R13(d13C_frac)
	
	return t0_frac, tf_frac, mass_frac, R13_frac, Fm_frac

#define a function to find the best-fit 13C/12C ratio for each peak.
def _fit_R13_peak(R13_frac, ind_wgh, lt, ec):
	'''
	Fits the 13C/12C ratio of each peak using DEa values in ``ec``.
	Called by ``IsotopeResult.__init__()``.
	
	Parameters
	----------
	R13_frac : np.ndarray
		Array of 13C/12C ratios for each fraction, length nFrac.

	ind_wgh : np.ndarray
		Index in ``lt.t`` corresponding to the mass-weighted mean time for
		each fraction. Length nFrac.

	lt : rp.LapalceTransform)
		``rp.LaplaceTransform`` instance containing the Laplace transform 
		matrix to convert f(Ea) peaks to carbon degradation rates at each 
		timepoint.

	ec : rp.EnergyComplex)
		``rp.EnergyComplex`` instance containing f(Ea) Gaussian peaks.

	Returns
	-------
	d13C_peak : np.ndarray
		Best-fit peak 13C/12C ratios for each peak as determined by
		``scipy.optimize.least_squares()`` and converted to d13C VPDB scale.

	d13C_rmse : float
		Fitting RMSE determined as ``norm(Ax-b)/sqrt(nFrac)``, and converted
		to d13C VPDB scale.

	Warnings
	--------
	Raises warning if ``scipy.optimize.least_squares`` cannot converge on a
	best-fit solution.
	'''
	
	#make initial guess of 0 per mille
	_,nPeak = np.shape(ec.peaks)
	nFrac = len(R13_frac)
	
	Rpdb = 0.011237
	r0 = Rpdb*np.ones(nPeak)

	#perform fit
	res = least_squares(_R13_diff,r0,
		bounds=(0,np.inf),
		args=(R13_frac, ind_wgh, lt, ec))

	#ensure success
	if not res.success:
		warnings.warn('R13 peak calc. could not converge on a successful fit')

	#best-fit result
	R13_peak = res.x
	d13C_peak = _R13_to_d13C(R13_peak)

	#calculate predicted R13 of each fraction and convert to d13C
	R13_frac_pred = res.fun + R13_frac
	d13C_frac = _R13_to_d13C(R13_frac)
	d13C_frac_pred = _R13_to_d13C(R13_frac_pred)

	#calculate RMSE
	d13C_rmse = norm(d13C_frac - d13C_frac_pred)/(nFrac**0.5)

	return (d13C_peak, d13C_rmse)

#define a function to calculate true - predicted difference
def _R13_diff(R13_peak, R13_frac, ind_wgh, lt, ec):
	'''
	Calculates the difference between measured and predicted 13C/12C ratio. 
	To be used by ``scipy.optimize.least_squares``.
	Called by ``_fit_R13_peak()``.

	Parameters
	----------
	R13_peak : np.ndarray
		13C/12C ratio for each peak, length nPeaks.

	R13_frac : np.ndarray
		Array of 13C/12C ratios for each fraction, length nFrac.

	ind_wgh : np.ndarray
		Index in ``lt.t`` corresponding to the mass-weighted mean time for
		each fraction. Length nFrac.

	lt : rp.LapalceTransform)
		``rp.LaplaceTransform`` instance containing the Laplace transform 
		matrix to convert f(Ea) peaks to carbon degradation rates at each 
		timepoint.

	ec : rp.EnergyComplex)
		``rp.EnergyComplex`` instance containing f(Ea) Gaussian peaks.

	Returns
	-------
	R13_diff : np.ndarray
		Difference between measured and predicted 13C/12C ratio for each 
		fraction, length nFrac.
	'''

	R13_CO2 = _calc_R13_CO2(R13_peak, lt, ec)

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

## PUBLIC FUNCTIONS ##

#define a function to blank-correct fraction isotopes
def blank_correct(t0_frac, tf_frac, mass_frac, R13_frac, Fm_frac):
	'''
	Performs blank correction (NOSAMS RPO instrument) on raw isotope values.
	Called by ``IsotopeResult.__init__()``.

	Parameters
	----------
	t0_frac : np.ndarray
		Array of t0 for each fraction (in seconds), length nFrac.

	tf_frac : np.ndarray
		Array of tf for each fraction (in seconds), length nFrac.

	mass_frac : np.ndarray
		Array of masses (ugC) for each fraction, length nFrac.
	
	R13_frac : np.ndarray
		Array of 13C/12C ratios for each fraction, length nFrac.
	
	Fm_frac : np.ndarray
		Array of Fm values for each fraction, length nFrac.

	Returns
	-------
	mass_frac_corr : np.ndarray
		Array of blank-corrected masses (ugC) for each fraction, length nFrac.
	
	R13_frac_corr : np.ndarray
		Array of blank-corrected 13C/12C ratios for each fraction, length 
		nFrac.
	
	Fm_frac_corr : np.ndarray
		Array of blank-corrected Fm values for each fraction, length nFrac.

	See Also
	--------
	IsotopeResult
		Class to perform isotope deconvolution and store results.
	
	References
	----------
	J.D. Hemingway et al. **(in prep)** Assessing the blank carbon
	contribution, isotope mass balance, and kinetic isotope fractionation of 
	the ramped pyrolysis/oxidation instrument at NOSAMS.
	'''

	#define constants
	bl_flux = 0.375/1000 #ug/s
	bl_Fm = 0.555
	bl_d13C = -29.0
	bl_R13 = _d13C_to_R13(bl_d13C) #converted to 13C/12C ratio

	#calculate blank mass for each fraction
	dt = tf_frac - t0_frac
	bl_mass = bl_flux*dt #ug

	#perform blank correction
	mass_frac_corr = mass_frac - bl_mass
	R13_frac_corr = (mass_frac*R13_frac - bl_mass*bl_R13)/mass_frac_corr
	Fm_frac_corr = (mass_frac*Fm_frac - bl_mass*bl_Fm)/mass_frac_corr

	return mass_frac_corr, R13_frac_corr, Fm_frac_corr


class IsotopeResult(object):
	__doc__='''
	Class for performing isotope deconvolution and storing results.

	Parameters
	----------
	sum_data : str or pd.DataFrame
		File containing isotope data, either as a path string or 
		``pd.DataFrame`` instance.

	lt : rp.LapalceTransform)
		``rp.LaplaceTransform`` instance containing the Laplace transform 
		matrix to convert f(Ea) peaks to carbon degradation rates at each 
		timepoint.

	ec : rp.EnergyComplex)
		``rp.EnergyComplex`` instance containing f(Ea) Gaussian peaks.

	blk_corr : boolean
		Tells the program whether or not to blank-correct isotope and mass
		data for each measured CO2 fraction. If `True`, corrects for the NOSAMS
		instrument blank as determined by Hemingway et al. **(in prep)**
		Defaults to `False`.

	mass_rsd : float
		Relative standard deviation on fraction masses. Defaults to 0.01 (i.e.
		1 percent of measured mass).

	add_noise : boolean
		Tells the program whether or not to add Gaussian noise to isotope and
		mass values. To be used for Monte Carlo uncertainty calculations. 
		Defaults to `False`.

	Raises
	------
	ValueError
		If nPeaks in the ``rp.EnergyComplex`` instance *(after `combine_last`*
		*has been applied!)* is greater than nFrac in `sum_data`. The problem
		is underconstrained and cannot be solved uniquely. Increase 
		`combine_last` or `omega` within the ``rp.EnergyComplex`` instance.

	ValueError
		If `sum_data` is not str or ``pd.DataFrame``.
	
	ValueError
		If `sum_data` does not contain "d13C", "d13C_std", "Fm", "Fm_std", 
		"ug_frac", and "fraction" columns.
	
	ValueError
		If index is not ``pd.DatetimeIndex`` instance.
	
	ValueError
		If first two rows are not fractions "-1" and "0"

	Warnings
	--------
	Raises warning if ``scipy.optimize.least_squares`` cannot converge on a
	best-fit solution.

	Notes
	-----
	For bookkeeping purposes, the first 2 rows must be fractions "-1" and "0",
	where the timestamp for fraction "-1" is the first point in `all_data` and
	the timestamp for fraction "0" is the t0 for the first fraction.

	This class uses ``rp.EnergyComplex`` peaks **after** the "comnine_last" 
	flag has been implemented. That is, it treats combined peaks as a single 
	peak when calculating tmax, Tmax, and isotope values for each peak.

	For d13C calculation, the Kinetic Isotope Effect (KIE) is accounted for
	using the `DEa` values in the ``rp.EnergyComplex`` instance. See Cramer et
	al. (2001) and Cramer (2004) for further discussion on the KIE in general,
	and Hemingway et al. **(in prep)** for discussion on the KIE with respect
	to the NOSAMS Ramped PyrOx instrument.

	Mass RMSE is probably a combination of the fact that true masses are
	measured offline as well as error from discretizing. the Sum of predicted
	fraction contributions is never perfectly equal to unity. Increasing nT
	lowers mass RMSE, but never to zero.

	See Also
	--------
	rampedpyrox.LaplaceTransform
		Instance of this class required as a parameter.

	rampedpyrox.EnergyComplex
		Instance of this class required as a parameter.

	blank_correct
		Method for blank-correcting CO2 fraction isotopes for the NOSAMS
		Ramped PyrOx instrument.

	Examples
	--------
	Fitting isotopes to an ``rp.EnergyComplex`` instance (ec) using a
	``rp.LaplaceTransform`` instance (lt)::

		#import data
		data = '/path_to_folder_containing_data/data.csv'

		#perform isotope regression
		ir = rp.IsotopeResult(data,lt, ec,
			blank_correct=True,
		 	mass_rsd=0.01,
		 	add_noise=True)

	Returning peak summary data from ``rp.IsotopeResult`` instance (ir)::

		#print summary
		ir.summary()

 	Attributes
 	----------
 	fraction_info : pd.DataFrame
 		Contains the t0 (seconds), tf (seconds), mass (ugC), d13C (VPDB), and
 		Fm of each measured CO2 fraction.

 	peak_info : pd.DataFrame
 		Contains the estimated mass (ugC), d13C (VPDB), and Fm of each peak
 		using ``scipy.optimize.least_squares`` (including KIE, for d13C) and
 		``scipy.optimize.nnls`` (for Fm).

 	RMSEs : np.ndarray
 		Array containing the mass, d13C, and Fm RMSEs for the fitting model.

	References
	----------
	\B. Cramer et al. (2001) Reaction kinetics of stable carbon isotopes in
	natural gas -- Insights from dry, open system pyrolysis experiments.
	*Energy & Fuels*, **15**, 517-532.

	\B. Cramer (2004) Methane generation from coal during open system 
	pyrolysis investigated by isotope specific, Gaussian distributed reaction
	kinetics. *Organic Geochemistry*, **35**, 379-392.

	J.D. Hemingway et al. **(in prep)** Assessing the blank carbon
	contribution, isotope mass balance, and kinetic isotope fractionation of 
	the ramped pyrolysis/oxidation instrument at NOSAMS.
	'''

	def __init__(self, sum_data, lt, ec, 
		blk_corr=False, mass_rsd=0.01, add_noise=False):

		#extract isotopes and time for each fraction
		t0_frac, tf_frac, mass_frac, R13_frac, Fm_frac = _extract_isotopes(
			sum_data, 
			mass_rsd=mass_rsd,
			add_noise=add_noise)

		#blank correct if necessary
		if blk_corr:
			mass_frac, R13_frac, Fm_frac = blank_correct(
				t0_frac, tf_frac, mass_frac, R13_frac, Fm_frac)

		#combine into pd.DataFrame and save as attribute
		nFrac = len(t0_frac)
		d13C_frac = _R13_to_d13C(R13_frac) #convert to d13C for storing
		frac_info = pd.DataFrame(np.column_stack((t0_frac, tf_frac, mass_frac,
			d13C_frac, Fm_frac)), columns=['t0 (s)','tf (s)','mass (ugC)',\
			'd13C','Fm'], index=np.arange(1,nFrac+1))

		self.fraction_info = frac_info

		#calculate peak contribution and indices of each fraction
		cont_ptf, ind_min, ind_max, ind_wgh = _calc_cont_ptf(
			lt, ec, t0_frac, tf_frac)

		#calculate peak masses, predicted fraction masses, and rmse
		#generate modeled thermogram and extract total mass (ugC)
		ugC = np.sum(mass_frac)
		g = np.inner(lt.A,ec.phi_hat) #fraction
		tg = -np.gradient(g)

		#calculate the mass of each peak in ugC
		mass_peak = ec.rel_area*ugC

		#calculate the predicted mass of each fraction in ugC
		mass_pred = []
		for imi,ima in zip(ind_min,ind_max):
			mass_pred.append(np.sum(tg[imi:ima+1])*ugC)
		
		mass_frac_pred = np.array(mass_pred)
		mass_rmse = norm(mass_frac_pred - mass_frac)/(nFrac**0.5)

		#perform R13 regression to calculate peak R13 values
		res_13 = _fit_R13_peak(R13_frac, ind_wgh, lt, ec)
		d13C_peak = res_13[0]
		d13C_rmse = res_13[1]

		#perform Fm regression
		res_14 = nnls(cont_ptf,Fm_frac)
		Fm_peak = res_14[0]
		Fm_rmse = res_14[1]/(nFrac**0.5)

		#repeat isotopes for combined peaks if necessary and append to arrays
		# makes book-keeping easier later, since we'll recomine all peak info.
		# for the summary tables
		nP_tot = len(mass_peak)
		_,nP_comb = np.shape(ec.peaks)
		d13C_peak = np.append(d13C_peak, d13C_peak[-1]*
			np.ones(nP_tot - nP_comb))
		Fm_peak = np.append(Fm_peak, Fm_peak[-1]*np.ones(nP_tot - nP_comb))

		#combine into pd.DataFrame and save as attribute
		peak_info = pd.DataFrame(np.column_stack((mass_peak, d13C_peak,
			Fm_peak)), columns=['mass (ugC)','d13C','Fm'],
			index=np.arange(1,nP_tot+1))

		self.peak_info = peak_info

		#store pd.Series of rmse values
		rmses = pd.Series([mass_rmse, d13C_rmse, Fm_rmse],
			index=['mass','d13C','Fm'])
		self.RMSEs = rmses

	def summary(self):
		'''
		Prints a summary of the ``rp.IsotopeResult`` instance.
		'''

		#define strings
		title = self.__class__.__name__ + ' summary table:'
		line = '=========================================================='
		fi = 'Isotopes and masses for each fraction:'
		pi = 'Isotope and mass estimates for each deconvolved peak:'
		note = 'NOTE: Combined peak results are repeated in summary table!'

		print(title + '\n\n' + line + '\n' + fi + '\n')
		print(self.fraction_info)
		print('\n' + line + '\n' + pi + '\n\n' + note + '\n')
		print(self.peak_info)
		print('\n' + line)
