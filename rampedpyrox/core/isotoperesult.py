'''
This module contains the IsotopeResult class for calculating the isotope
composition of individual Ea peaks within a sample, as well as supporting
functions.

Note:
	mass RMSE is probably a combination of the fact that true masses are
	measured offline as well as error from discretizing -- thermograms
	appear to over-predict manometrically-calculated masses. Increasing nT 
	lowers mass RMSE, but never to zero.


* TODO: fix _fir_R13_peak to handle combined peaks.
'''

import numpy as np
import pandas as pd

from numpy.linalg import norm
from scipy.optimize import least_squares
from scipy.optimize import nnls

from rampedpyrox.core.energycomplex import _phi_hat

def _calc_R13_CO2(R13_peak, DEa, ec, lt):
	'''
	Performs a best-fit for 13C ratios, including DEa values.
	
	Args:
		R13_peak (np.ndarray): 13C/12C ratio for each peak.

		DEa (int, float, or np.ndarray): DEa values, either a scalar or vector
			of length ec.mu. DEa in units of kJ!

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

		DEa (int, float, or np.ndarray): DEa values, either a scalar or vector
			of length ec.mu. DEa in units of kJ!

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
	Fits the 13C/12C of each peak using inputted DEa values for each peak.
	
	Args:
		R13_frac (np.ndarray): 13C/12C ratio for each fraction. Length nFrac.

		frac_ind (np.ndarray): Index of mass-weighted mean for each fraction.
			Length nFrac.

		DEa (int, float, or np.ndarray): DEa values, either a scalar or vector
			of length ec.mu. DEa in units of kJ!

		ec (rp.EnergyComplex): Energy complex object containing peaks.

		lt (rp.LaplaceTransform): Laplace transform to forward-model 
			isotope-specific thermograms.

	Returns:
		R13_peak (np.ndarray): Best-fit peak 13C/12C ratios as determined by
			``scipy.optimize.least_squares()``.

	'''
	
	#make initial guess of 0 per mille
	r0 = 0.011237*np.ones(len(ec.mu))

	#perform fit
	res = least_squares(_R13_diff,r0,
		bounds=(0,np.inf),
		args=(R13_frac,frac_ind,DEa,ec,lt))

	R13_peak = res.x

	return R13_peak

#GOOD
def _blank_correct(t0_frac, tf_frac, mass_frac, R13_frac, Fm_frac):
	'''
	Performs blank correction (NOSAMS RPO instrument) on raw isotope values.

	Args:
		t0_frac (np.ndarray): Array of t0 for each fraction, length nF.

		tf_frac (np.ndarray): Array of tf for each fraction, length nF.

		mass_frac (np.ndarray): Array of masses (ugC) for each fraction, 
			length nF.
		
		R13_frac (np.ndarray): Array of 13R values for each fraction, 
			length nF.
		
		Fm_frac (np.ndarray): Array of Fm values for each fraction, length nF.

	Returns:
		mass_frac_corr (np.ndarray): Array of masses (ugC) for each fraction, 
			length nF. Corrected for blank contribution.
		
		R13_frac_corr (np.ndarray): Array of 13R values for each fraction, 
			length nF. Corrected for blank contribution.
		
		Fm_frac_corr (np.ndarray): Array of Fm values for each fraction, 
			length nF. Corrected for blank contribution.
	
	References:
		J.D. Hemingway et al. (2016) Assessing the blank carbon contribution,
			isotope mass balance, and kinetic isotope fractionation of the
			ramped pyrolysis/oxidation instrument at NOSAMS. *Radiocarbon*,
			**(in prep)**.
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

#GOOD
def _calc_cont_ptf(lt, ec, t0_frac, tf_frac):
	'''
	Calculates the contribution of each peak to each fraction.
	Called by ``IsotopeResult.__init__()``.

	Args:
		lt (rp.LapalceTransform): ``LaplaceTransform`` object containing the
			Laplace transform matrix to convert EC peaks to carbon degradation
			rates at each timepoint.

		ec (rp.EnergyComplex): ``EnergyComplex`` object containing EC peaks
			(both for 12C and 13C).

		t0_frac (np.ndarray): Array of t0 for each fraction, length nF.

		tf_frac (np.ndarray): Array of tf for each fraction, length nF.

	Returns:
		cont_ptf (np.ndarray): 2d array of the contribution by each peak to
			each measured CO2 fraction with shape [nFrac x nPeak].

		ind_min (np.ndarray): Minimum index for each fraction. Length nFrac.

		ind_max (np.ndarray): Maximum index for each fraction. Length nFrac.

		ind_wgh (np.ndarray): Index of mass-weighted mean for each fraction.
			Length nFrac.

	Raises:
		ValueError: If nPeaks >  nFrac, the problem is underconstrained.

	Notes:
		This function uses EC peaks **after** the "comnine_last" flag has been
		implemented. That is, it treats combined peaks as a single peak when
		calculating indices and contributions to each fraction.
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
	peaks = -np.gradient(g_peak, axis=0) #fraction/timestep (each peak)

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

#GOOD
def _d13C_to_R13(d13C):
	'''
	Converts d13C values to 13R values using VPDB standard.

	Args:
		d13C (np.ndarray): Inputted d13C values.

	Returns:
		R13 (np.ndarray): d13C values converted to 13C ratios.
	'''

	Rpdb = 0.011237 #13C/12C ratio VPDB

	R13 = (d13C/1000 + 1)*Rpdb

	return R13

#GOOD
def _extract_isotopes(sum_data, mass_rsd=0, add_noise=False):
	'''
	Extracts isotope data from the "sum_data" file.

	Args:
		sum_data (str or pd.DataFrame): File containing isotope data,
			either as a path string or pandas.DataFrame object.

		mass_rsd (float): Relative standard deviation on fraction masses.
			Defaults to 0.01 (i.e. 1%).

		add_noise (boolean): Tells the program whether or not to add Gaussian
			noise to isotope and mass values. To be used for Monte Carlo
			uncertainty calculations. Defaults to False.

	Returns:
		t0_frac (np.ndarray): Array of t0 for each fraction, length nF.

		tf_frac (np.ndarray): Array of tf for each fraction, length nF.

		mass_frac (np.ndarray): Array of masses (ugC) for each fraction, 
			length nF.
		
		R13_frac (np.ndarray): Array of 13R values for each fraction, 
			length nF.
		
		Fm_frac (np.ndarray): Array of Fm values for each fraction, length nF.


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

#GOOD
def _R13_to_d13C(R13):
	'''
	Converts 13R values to d13C values using VPDB standard.

	Args:
		R13 (np.ndarray): d13C values converted to 13C ratios.

	Returns:
		d13C (np.ndarray): Inputted d13C values.
	'''

	Rpdb = 0.011237 #13C/12C ratio VPDB

	d13C = (R13/Rpdb - 1)*1000

	return d13C


class IsotopeResult(object):
	'''
	Class for performing isotope deconvolution
	'''

	def __init__(self, sum_data, lt, ec, 
		blank_correct=False, mass_rsd=0, add_noise=False):

		#extract isotopes and time for each fraction
		t0_frac, tf_frac, mass_frac, R13_frac, Fm_frac = _extract_isotopes(
			sum_data, 
			mass_rsd=mass_rsd,
			add_noise=add_noise)

		#blank correct if necessary
		if blank_correct:
			mass_frac, R13_frac, Fm_frac = _blank_correct(
				t0_frac, tf_frac, mass_frac, R13_frac, Fm_frac)

		#combine into pd.DataFrame and save as attribute
		nFrac = len(t0_frac)
		d13C_frac = _R13_to_d13C(R13_frac) #convert to d13C for storing
		frac_info = pd.DataFrame(np.column_stack((t0_frac, tf_frac, mass_frac,
			d13C_frac, Fm_frac)), columns=['t0','tf','mass (ugC)','d13C','Fm'],
			index=np.arange(1,nFrac+1))

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

		#perform R13 regression
		d13C_rmse = 0

		#perform Fm regression
		res = nnls(cont_ptf,Fm_frac)
		Fm_peak = res[0]
		Fm_rmse = res[1]/(nFrac**0.5)

		#combine into pd.DataFrame and save as attribute
		# nPeak = len(mass_peak)
		# d13C_peak = _R13_to_d13C(R13_peak) #convert to d13C for storing
		# peak_info = pd.DataFrame(np.column_stack((mass_peak, d13C_peak,
		# 	Fm_peak)), columns=['mass (ugC)','d13C','Fm'],
		# 	index=np.arange(1,nPeak+1))

		# self.peak_info = peak_info

		#store pd.Series of rmse values
		rmses = pd.Series([mass_rmse,d13C_rmse,Fm_rmse],
			index=['mass','d13C','Fm'])
		self.RMSEs = rmses

		# #calculate predicted Fm for each fraction and store difference
		# Fm_pred = np.inner(cont_ptf,self.Fm_peak)
		# self.Fm_pred_meas = Fm_pred - self.Fm_frac

		# #calculate predicted 13C/12C ratio including DEa
		# R13_peak = _fit_R13_peak(R13[:,0], frac_ind, DEa, ec, lt)

		# #convert to d13C and store
		# d13C_peak,_ = _13R_to_d13C(R13_peak,0)
		# self.d13C_peak = d13C_peak



		# #perform 13R regression
		# if DEa is None:
		# 	#no DEa, perform nnls on raw data
		# 	R13_peak = nnls(cont_ptf,R13[:,0])[0]
		# 	d13C_peak,_ = _13R_to_d13C(R13_peak, 0)
		# 	self.d13C_peak = d13C_peak

		# #calculate difference and store
		# R13_pred = np.inner(cont_ptf,R13_peak)
		# d13C_pred,_ = _13R_to_d13C(R13_pred,0)
		# self.d13C_pred_meas = d13C_pred - self.d13C_frac
		


	def summary():
		'''
		Prints a summary of the IsotopeResult.
		'''