'''
This module contains helper functions for the model classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_calc_cmpt', '_calc_f', '_calc_R', '_rpo_calc_A']

import numpy as np

from numpy.linalg import norm
from scipy.optimize import nnls

#import helper functions
from ..core.core_functions import(
	assert_len,
	)

#define a function to generate estimated time data from model and ratedata
def _calc_cmpt(model, ratedata):
	'''
	Calculates the timedata for a given ``rp.RateData`` and ``rp.Model`` 
	instance.

	Parameters
	----------
	model : rp.Model
		The ``rp.Model`` instance used to calculate the forward model.

	ratedata : rp.RateData
		The ``rp.RateData`` instance containing the reactive continuum data.

	Returns
	-------
	cmpt : np.ndarray
		A 2d array of the fraction of each component remaining at each
		timepoint. Shape [`nt` x `nPeak`] (after combining).
	'''

	return np.inner(model.A, ratedata.peaks.T)

#define a function to generate estimated rate data from model and timedata
def _calc_f(model, timedata, omega):
	'''
	Calculates the reactive continuum of rates (or Ea, for DAEM) for a given
	``rp.TimeData`` and ``rp.Model`` instance.

	Parameters
	----------
	model : rp.Model
		``rp.Model`` instance containing the A matrix to use for calculation.

	timedata : rp.TimeData
		``rp.Timedata`` instance containing the fraction remaining with time 
		array to use for the calculation.

	omega : scalar
		Tikhonov regularization weighting factor.

	Returns
	-------
	f : np.ndarray
		Array of the pdf of the discretized distribution of rates (or Ea,
		for DAEM).

	resid_rmse : float
		Residual RMSE between true and modeled time data.

	rgh_rmse : float
		Roughness RMSE from Tikhonov Regularization.

	References
	----------
	[1] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.

	[2] P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
		Numerical aspects of linear inversion (monographs on mathematical
		modeling and computation). *Society for Industrial and Applied*
		*Mathematics*.

	[3] P.C. Hansen (1994) Regularization tools: A Matlab package for analysis
		and solution of discrete ill-posed problems. *Numerical Algorithms*, 
		**6**, 1-35.
	'''

	#extract nt and nk (or nEa for daem)
	nt, nk = np.shape(model.A)

	#calculate the regularization matrix
	R = _calc_R(nk)

	#concatenate A+R and g+zeros
	A_reg = np.concatenate(
		(model.A, R*omega))

	g_reg = np.concatenate(
		(timedata.g, np.zeros(nk+1)))

	#calculate inverse results and estimated g
	f, _ = nnls(A_reg, g_reg)
	g_hat = np.inner(model.A, f)
	rgh = np.inner(R, f)

	#calculate errors
	resid_rmse = norm(timedata.g - g_hat)/nt**0.5
	rgh_rmse = norm(rgh)/nk**0.5

	return f, resid_rmse, rgh_rmse

#define a function to calculate the Tikhonov regularization matrix
def _calc_R(n):
	'''
	Calculates regularization matrix (`R`) for a given size, n.

	Parameters
	----------
	n : int
		Size of the regularization matrix will be [`n+1` x `n`].

	Returns
	-------
	R : np.ndarray
		Regularization matrix of size [`n+1` x `n`].

	References
	----------
	[1] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.

	[2] P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
		Numerical aspects of linear inversion (monographs on mathematical
		modeling and computation). *Society for Industrial and Applied*
		*Mathematics*.

	[3] P.C. Hansen (1994) Regularization tools: A Matlab package for analysis
		and solution of discrete ill-posed problems. *Numerical Algorithms*, 
		**6**, 1-35.
	'''

	R = np.zeros([n+1, n])

	#ensure pdf = 0 outside of Ea range specified
	R[0, 0] = 1.0
	R[-1, -1] = -1.0

	#1st derivative operator
	c = [-1, 1]

	#populate matrix
	for i, row in enumerate(R):
		if i != 0 and i != n:
			row[i - 1:i + 1] = c

	return R

#define function to calculte the A matrix for DAEM models
def _rpo_calc_A(Ea, log10k0, t, T):
	'''
	Calculates the A matrix for a DAEM model (e.g. a Ramped Pyrox run).

	Parameters
	----------
	Ea : array-like
		Array of activation energy points to be used in the A matrix, in kJ.
		Length `nEa`.

	log10k0 : scalar or array-like
		Arrhenius pre-exponential factor, either a constant value, array with
		length `nEa`, or a lambda function of Ea.

	t : array-like
		Array of timepoints to be used in the A matrix, in seconds. Length `nt`.

	T : array-like
		Array of temperature to be used in the A matrix, in Kelvin. Length `nt`.

	Returns
	-------
	A : np.ndarray
		2d array of the Laplace transform for the Daem model. 
		Shape [`nt` x `nEa`].

	References
	----------
	[1] R.L Braun and A.K. Burnham (1987) Analysis of chemical reaction 
		kinetics using a distribution of activation energies and simpler 
		models. *Energy & Fuels*, **1**, 153-161.

	[2] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.
	'''

	#set constants
	nt = len(t)
	nEa = len(Ea)
	R = 8.314/1000 #kJ/mol/K

	#get arrays in the right format and ensure lengths
	Ea = assert_len(Ea, nEa) #kJ
	t = assert_len(t, nt) #s
	T = assert_len(T, nt) #K

	#get log10k0 into the right format
	if hasattr(log10k0,'__call__'):
		log10k0 = log10k0(Ea)
	
	log10k0 = assert_len(log10k0, nEa) 
	k0 = 10**log10k0 #s-1

	#calculate time and Ea gradients
	dt = np.gradient(t)
	dEa = np.gradient(Ea)

	#pre-allocate A
	A = np.zeros([nt,nEa])

	#loop through each timestep
	for i,_ in enumerate(T):

		#generate arrays
		U = T[:i] #Kelvin, [1,i]
		dtau = dt[:i] #seconds, [1,i]

		#generate matrices
		eps_mat = np.outer(Ea, np.ones(i)) #kJ, [nEa,i]
		k0_mat = np.outer(k0, np.ones(i)) #s-1, [nEa,i]
		u_mat = np.outer(np.ones(nEa), U) #Kelvin, [nEa,i]

		#generate A for row i (i.e. for each timepoint) and store in A
		hE_mat = -k0_mat*dtau*np.exp(-eps_mat/(R*u_mat)) #unitless, [nEa,i]
		A[i] = np.exp(np.sum(hE_mat, axis = 1))*dEa #kJ

	return A
