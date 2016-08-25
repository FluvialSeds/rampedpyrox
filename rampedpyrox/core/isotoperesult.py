'''
This module contains the IsotopeResult class for calculating the isotope
composition of individual Ea peaks within a sample, as well as supporting
functions.

* TODO: update _calc_cont_ptf to handle fractions of timepoints
* TODO: fix blank correction d13C stdev.
'''

import numpy as np
import pandas as pd

from scipy.optimize import least_squares
from scipy.optimize import nnls

from rampedpyrox.core.energycomplex import _phi_hat

def _calc_R13_CO2(R13_peak, DEa, ec, lt):
	'''
	Performs a best-fit for 13C ratios, including ∆Ea values.
	
	Args:
		R13_peak (np.ndarray): 13C/12C ratio for each peak.

		DEa (int, float, or np.ndarray): ∆Ea values, either a scalar or vector
			of length ec.mu. ∆Ea in units of kJ!

		ec (rp.EnergyComplex): Energy complex object containing peaks.

		lt (np.LaplaceTransform): Laplace transform to forward-model 
			isotope-specific thermograms.

	Returns:
		R13_CO2 (np.ndarray): Array of 13C/12C ratio of instantaneously eluted
			CO2 at each timepoints with length nT.

	Raises:
		ValueError: If DEa is not int, float, or np.ndarray.

		ValueError: If DEa is np.ndarray and is of different length than ec.mu

		ValueError: If R13C_peak is of different length than ec.mu

	'''

	#check DEa
	if not isinstance(DEa, (float,int,np.ndarray)):
		raise ValueError('DEa must be float, int, or np.ndarray')
	elif isinstance(DEa, np.ndarray) and len(DEa) is not len(ec.mu):
		raise ValueError('If array, DEa must have same length as ec.mu')

	#check R13_peak
	if not isinstance(R13_peak, np.ndarray) or len(R13_peak) is not len(ec.mu):
		raise ValueError('R13_peak must be np.ndarray with same length as ec.mu')

	eps = ec.eps
	
	#extract 12C data
	mu = ec.mu
	sigma = ec.sigma
	height = ec.height/(1+R13_peak) #convert total height to 12C

	#calculate 13C data
	mu_13 = mu + DEa
	sigma_13 = sigma
	height_13 = height*R13_peak

	#generate 13C and 12C f(Ea) distributions
	phi_hat_12,_ = _phi_hat(eps, mu, sigma, height)
	phi_hat_13,_ = _phi_hat(eps, mu_13, sigma_13, height_13)

	#forward-model 13C and 12C g_hat
	g_hat_12 = np.inner(lt.A,phi_hat_12)
	g_hat_13 = np.inner(lt.A,phi_hat_13)

	#convert to 13C and 12C thermograms, and calculate R13_CO2
	grad_t = np.gradient(lt.t)
	gdot_hat_12 = -np.gradient(g_hat_12)/grad_t
	gdot_hat_13 = -np.gradient(g_hat_13)/grad_t

	R13_CO2 = gdot_hat_13/gdot_hat_12

	return R13_CO2

def _R13_diff(R13_peak, R13_frac, frac_ind, DEa, ec, lt):
	'''
	Function to calculate the difference between measured and predicted 13C/12C
	ratio. To be used by ``scipy.optimize.least_squares``.
	Called by ``_fit_R13_peak``.

	Args:
		R13_peak (np.ndarray): 13C/12C ratio for each peak. Length nPeaks.

		R13_frac (np.ndarray): 13C/12C ratio for each fraction. Length nFrac.

		frac_ind (np.ndarray): Index of mass-weighted mean for each fraction.
			Length nFrac.

		DEa (int, float, or np.ndarray): ∆Ea values, either a scalar or vector
			of length ec.mu. ∆Ea in units of kJ!

		ec (rp.EnergyComplex): Energy complex object containing peaks.

		lt (np.LaplaceTransform): Laplace transform to forward-model 
			isotope-specific thermograms.

	Returns:
		R13_diff (np.ndarray): Difference between measured and predicted 13C/12C
			ratio for each fraction. Length nFrac.
	'''

	R13_CO2 = _calc_R13_CO2(R13_peak, DEa, ec, lt)

	R13_diff = R13_CO2[frac_ind] - R13_frac

	return R13_diff

def _fit_R13_peak(R13_frac, frac_ind, DEa, ec, lt):
	'''
	Fits the 13C/12C of each peak using inputted ∆Ea values for each peak.
	
	Args:
		R13_frac (np.ndarray): 13C/12C ratio for each fraction. Length nFrac.

		frac_ind (np.ndarray): Index of mass-weighted mean for each fraction.
			Length nFrac.

		DEa (int, float, or np.ndarray): ∆Ea values, either a scalar or vector
			of length ec.mu. ∆Ea in units of kJ!

		ec (rp.EnergyComplex): Energy complex object containing peaks.

		lt (rp.LaplaceTransform): Laplace transform to forward-model 
			isotope-specific thermograms.

	Returns:
		R13_peak (np.ndarray): Best-fit peak 13C/12C ratios as determined by
			``scipy.optimize.least_squares()``.

	'''

	#calculate mass-weighted mean indices for each fraction
	frac_ind = _calc_frac_ind()
	
	#make initial guess of 0‰
	r0 = 0.011237*np.ones(len(R13_peak))

	#perform fit
	res = least_squares(_R13_diff,r0,
		bounds=(0,np.inf),
		args=(R13_frac,frac_ind,DEa,ec,lt))

	R13_peak = res.x

	return R13_peak

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

def _calc_cont_ptf(mod_tg, t):
	'''
	Calculates the contribution of each peak to each fraction.
	Called by ``IsotopeResult.__init__()``.

	Args:
		mod_tg (rp.ModeledData): ``ModeledData`` object containing peaks
			of interest for isotope deconvolution. 

		t (np.ndarray): 2d array of the t0 and tf (in seconds) for each
			fraction, with shape [nFrac x 2].

	Returns:
		cont_ptf (np.ndarray): 2d array of the contribution by each peak to
			each measured CO2 fraction with shape [nFrac x nPeak].

		frac_ind (np.ndarray): Index of mass-weighted mean for each fraction.
			Length nFrac.

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

	#pre-allocate cont_ptf matrix and frac_ind array
	cont_ptf = np.zeros([nF,nP])
	frac_ind = np.zeros(nF)

	#loop through and calculate contributions
	for i,row in enumerate(t):
		#extract indices for each fraction
		ind = np.where((md_t >= row[0]) & (md_t <= row[1]))

		#calculate peak to fraction contribution
		ptf_i = np.sum(peaks[ind],axis=0)/np.sum(tot[ind])

		#store in row i
		cont_ptf[i] = ptf_i

		#calculate mass-weighted fraction index
		av_t = np.average(md_t[ind],weights=md_tot[ind])
		frac_ind[i] = np.argmax(md_t >= av_t) #calculates first instance above

	return cont_ptf, frac_ind

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

	def __init__(self, sum_data, mod_tg, ec, lt, 
		blank_correct=False, DEa=0, mass_rsd=0.01):

		#extract isotopes and time
		t, R13, Fm, mass = _extract_isotopes(sum_data, mass_rsd=mass_rsd)

		#blank correct if necessary
		if blank_correct:
			R13,Fm,mass = _blank_correct(R13, Fm, t, mass)

		d13C, d13C_std = _13R_to_d13C(R13[:,0], R13[:,1])

		#calculate peak contribution to each fraction
		cont_ptf, frac_ind = _calc_cont_ptf(mod_tg, t)

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

		#calculate predicted 13C/12C ratio including ∆Ea
		R13_peak = _fit_R13_peak(R13[:,0], frac_ind, DEa, ec, lt)

		#convert to d13C and store
		d13C_peak,_ = _13R_to_d13C(R13_peak,0)
		self.d13C_peak = d13C_peak



		# #perform 13R regression
		# if DEa is None:
		# 	#no ∆Ea, perform nnls on raw data
		# 	R13_peak = nnls(cont_ptf,R13[:,0])[0]
		# 	d13C_peak,_ = _13R_to_d13C(R13_peak, 0)
		# 	self.d13C_peak = d13C_peak

		#calculate difference and store
		R13_pred = np.inner(cont_ptf,R13_peak)
		d13C_pred,_ = _13R_to_d13C(R13_pred,0)
		self.d13C_pred_meas = d13C_pred - self.d13C_frac
		


	def summary():
		'''
		Prints a summary of the IsotopeResult.
		'''