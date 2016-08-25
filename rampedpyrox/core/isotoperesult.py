'''
This module contains the IsotopeResult class for calculating the isotope
composition of individual Ea peaks within a sample, as well as supporting
functions.

* TODO: update _calc_cont_ptf to handle fractions of timepoints
* TODO: fix blank correction d13C stdev.
'''

import numpy as np
import pandas as pd

from scipy.optimize import nnls

def _blank_correct(R13, Fm, t, mass):
	'''
	Performs blank correction (NOSAMS RPO instrument) on raw isotope values.

	Args:
		R13 (np.ndarray): 2d array of 13R values, 1st column is mean, 2nd
			column is stdev.

		Fm (np.ndarray): 2d array of Fm values, 1st column is mean, 2nd
			column is stdev.

		t (np.ndarray): 2d array of the t0 and tf (in seconds) for each
			fraction, with shape [nFrac x 2].

		mass (np.ndarray): 2d array of masses (ugC), 1st column is mean, 2nd
			column is stdev.

	Returns:
		R13_corr (np.ndarray): 2d array of corrected 13R values, 1st column 
			is mean, 2nd column is stdev.

		Fm_corr (np.ndarray): 2d array of corrected Fm values, 1st column is 
			mean, 2nd column is stdev.

		mass_corr (np.ndarray): 2d array of corrected masses (ugC), 1st 
			column is mean, 2nd column is stdev.
	
	References:
		J.D. Hemingway et al. (2016) Assessing the blank carbon contribution,
			isotope mass balance, and kinetic isotope fractionation of the
			ramped pyrolysis/oxidation instrument at NOSAMS. *Radiocarbon*,
			**(in prep)**.
	'''

	#define constants
	bl_flux = 0.375/1000 #ug/s
	bl_flux_std = 0.058/1000 #ug/s
	bl_Fm = 0.555
	bl_Fm_std = 0.042
	bl_d13C = -29.0
	bl_d13C_std = 0.1

	#convert d13C to ratios
	bl_R13, bl_R13_std = _d13C_to_13R(bl_d13C, bl_d13C_std)

	#calculate blank mass for each fraction
	dt = t[:,1]-t[:,0]
	bl_mass = bl_flux*dt #ug
	bl_mass_std = bl_flux_std*dt #ug

	#perform mass blank correction
	sam_mass = mass[:,0] - bl_mass
	sam_mass_std = (mass[:,1]**2 + bl_mass_std**2)**0.5
	mass_corr = np.column_stack((sam_mass,sam_mass_std))

	#perform R13 blank correction
	sam_R13 = (mass[:,0]*R13[:,0] - bl_mass*bl_R13)/sam_mass
	sam_R13_std = (R13[:,1]**2 + \
		(bl_mass_std*bl_R13/sam_mass)**2 + \
		(bl_mass*bl_R13_std/sam_mass)**2 + \
		(bl_mass*bl_R13*sam_mass_std/(sam_mass**2))**2)**0.5
	R13_corr = np.column_stack((sam_R13, sam_R13_std))

	#perform Fm blank correction
	sam_Fm = (mass[:,0]*Fm[:,0] - bl_mass*bl_Fm)/sam_mass
	sam_Fm_std = (Fm[:,1]**2 + \
		(bl_mass_std*bl_Fm/sam_mass)**2 + \
		(bl_mass*bl_Fm_std/sam_mass)**2 + \
		(bl_mass*bl_Fm*sam_mass_std/(sam_mass**2))**2)**0.5
	Fm_corr = np.column_stack((sam_Fm, sam_Fm_std))

	return R13_corr, Fm_corr, mass_corr

def _calc_cont_ptf(mod_tg,t):
	'''
	Calculates the contribution of each peak to each fraction.
	Called by ``_fit()``.

	Args:
		mod_tg (rp.ModeledData): ``ModeledData`` object containing peaks
			of interest for isotope deconvolution. 

		t (np.ndarray): 2d array of the t0 and tf (in seconds) for each
			fraction, with shape [nFrac x 2].

	Returns:
		cont_ptf(np.ndarray): 2d array of the contribution by each peak to
			each measured CO2 fraction with shape [nFrac x nPeak].

	Raises:
		ValueError: If nPeaks >  nFrac, the problem is underconstrained.
	'''

	#extract data from modeled thermogram
	md_t = mod_tg.t
	md_p = mod_tg.gpdot_t
	md_tot = mod_tg.gdot_t

	nF,_ = np.shape(t)
	_,nP = np.shape(md_p)

	if nP > nF:
		raise ValueError('Under constrained problem! nPeaks > nFractions!!')

	#calculate areas by multiplying by time gradient (for variable timesetep)
	tot = md_tot*np.gradient(md_t)
	grad_mat = np.outer(np.gradient(md_t),np.ones(nP))
	peaks = md_p*grad_mat

	#pre-allocate cont_ptf matrix
	cont_ptf = np.zeros([nF,nP])

	#loop through and calculate contributions
	for i,row in enumerate(t):
		#extract indices for each fraction
		ind = np.where((md_t >= row[0]) & (md_t <= row[1]))
		ptf_i = np.sum(peaks[ind],axis=0)/np.sum(tot[ind])

		#store in row i
		cont_ptf[i] = ptf_i

	return cont_ptf

def _d13C_to_13R(d13C, d13C_std):
	'''
	Converts d13C values to 13R values using VPDB standard.

	Args:
		d13C (np.ndarray): Inputted d13C values.

		d13C_std (np.ndarray): Inputted d13C stdev.

	Returns:
		R13 (np.ndarray): d13C values converted to 13C ratios.

		R13_std (np.ndarray): d13C stdev. values converted to 13C ratios.
	'''

	Rpdb = 0.011237 #13C/12C ratio VPDB

	R13 = (d13C/1000 + 1)*Rpdb
	R13_std = Rpdb*d13C_std/1000

	return R13, R13_std

def _extract_isotopes(sum_data, mass_rsd=0.01):
	'''
	Extracts isotope data from the "sum_data" file.

	Args:
		sum_data (str or pd.DataFrame): File containing isotope data,
			either as a path string or pandas.DataFrame object.

		mass_rsd (float): Relative standard deviation on fraction masses.
			Defaults to 0.01 (i.e. 1%).

	Returns:
		t (np.ndarray): 2d array of times, 1st column is t0, 2nd column is tf
		
		R13 (np.ndarray): 2d array of 13R values, 1st column is mean, 2nd
			column is stdev.
		
		Fm (np.ndarray): 2d array of Fm values, 1st column is mean, 2nd
			column is stdev.

		mass (np.ndarray): 2d array of carbon mass (ug), 1st column is mean,
			2nd column is stdev.

	Raises:
		ValueError: If `sum_data` is not str or pd.DataFrame.
		
		ValueError: If `sum_data` does not contain "d13C", "d13C_std", "Fm",
			"Fm_std", "ug_frac", and "fraction" columns.
		
		ValueError: If index is not `DatetimeIndex`.
		
		ValueError: If first two rows are not fractions "-1" and "0"
	'''

	#import sum_data as a pd.DataFrame if inputted as a string path and check
	#that it is in the right format
	if isinstance(sum_data,str):
		sum_data = pd.DataFrame.from_csv(sum_data)
	elif not isinstance(sum_data,pd.DataFrame):
		raise ValueError('sum_data must be pd.DataFrame or path string')

	if 'fraction' and 'd13C' and 'd13C_std' and 'Fm' and 'Fm_std' and 'ug_frac' \
		not in sum_data.columns:
		raise ValueError('sum_data must have "fraction", "d13C", "d13C_std",'\
			' "Fm", "Fm_std", and "ug_frac" columns')

	if not isinstance(sum_data.index,pd.DatetimeIndex):
		raise ValueError('sum_data index must be DatetimeIndex')

	if sum_data.fraction[0] != -1 or sum_data.fraction[1] != 0:
		raise ValueError('First two rows must be fractions "-1" and "0"')

	#extract time data
	secs = (sum_data.index - sum_data.index[0]).seconds
	t0 = secs[1:-1]
	tf = secs[2:]
	t = np.column_stack((t0,tf))

	#extract d13C data and convert to ratio
	d13C_mean = sum_data.d13C[2:]
	d13C_std = sum_data.d13C_std[2:]
	R13_mean,R13_std = _d13C_to_13R(d13C_mean, d13C_std)
	R13 = np.column_stack((R13_mean,R13_std))

	#extract Fm data
	Fm_mean = sum_data.Fm[2:]
	Fm_std = sum_data.Fm_std[2:]
	Fm = np.column_stack((Fm_mean,Fm_std))

	#extract mass of each fraction, assume 1% rsd
	mass_mean = sum_data.ug_frac[2:]
	mass_std = mass_rsd*mass_mean
	mass = np.column_stack((mass_mean,mass_std))
	
	return t, R13, Fm, mass

def _fit(mod_tg, R13, Fm, t):
	'''
	Performs the fit to calculate peak isotope values.
	Called by ``IsotopeResult.__init__()``.

	Args:

	Returns:
	'''
	
	A = _calc_cont_ptf(mod_tg,t)

	#calculate Fm values
	Fm_peak = nnls(A,Fm[:,0])[0]

	#calculate d13C values
	R13_peak = nnls(A,R13[:,0])[0]
	d13C_peak = _13R_to_d13C(R13_peak)

	return d13C_peak, Fm_peak


def _13R_to_d13C(R13, R13_std):
	'''
	Converts 13R values to d13C values using VPDB standard.

	Args:
		R13 (np.ndarray): d13C values converted to 13C ratios.
		
		R13_std (np.ndarray): d13C stdev. values converted to 13C ratios.

	Returns:
		d13C (np.ndarray): Inputted d13C values.
		
		d13C_std (np.ndarray): Inputted d13C stdev.
	'''

	Rpdb = 0.011237 #13C/12C ratio VPDB

	d13C = (R13/Rpdb - 1)*1000
	d13C_std = 1000*R13_std/Rpdb

	return d13C, d13C_std


class IsotopeResult(object):
	'''
	Class for performing isotope deconvolution
	'''

	def __init__(self, sum_data, mod_tg, blank_correct=False, DEa=None, mass_rsd=0.01):

		#extract isotopes and time
		t, R13, Fm, mass = _extract_isotopes(sum_data, mass_rsd=mass_rsd)

		#blank correct if necessary
		if blank_correct:
			R13,Fm,mass = _blank_correct(R13, Fm, t, mass)

		d13C, d13C_std = _13R_to_d13C(R13[:,0], R13[:,1])

		#calculate peak contribution to each fraction
		cont_ptf = _calc_cont_ptf(mod_tg, t)

		#define public parameters
		self.t = t
		self.d13C_frac = d13C
		self.d13C_frac_std = d13C_std
		self.Fm_frac = Fm[:,0]
		self.Fm_frac_std = Fm[:,1]
		self.mass_frac = mass[:,0]
		self.mass_std = mass[:,1]

		#perform Fm regression and store data
		self.Fm_peak = nnls(cont_ptf,self.Fm_frac)[0]

		#calculate predicted Fm for each fraction and store difference
		Fm_pred = np.inner(cont_ptf,self.Fm_peak)
		self.Fm_pred_meas = Fm_pred - self.Fm_frac

		#perform 13R regression



		#d13C_peak,Fm_peak = _fit(mod_tg, R13, Fm, t)

		#self.d13C_peak = d13C_peak
		


	def summary():
		'''
		Prints a summary of the IsotopeResult.
		'''