'''
This module contains helper functions for the model classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_calc_ghat', '_calc_p', '_calc_R', '_rpo_calc_A']

import numpy as np

from numpy.linalg import norm
from scipy.optimize import nnls

#import helper functions
from .core_functions import(
	assert_len,
	)

#define a function to generate estimated time data from model and ratedata
def _calc_ghat(model, ratedata):
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
	ghat : array-like
		Array of estimated fraction of total carbon remaining at each 
		timestep. Length `nt`.
	'''

	return np.inner(model.A, ratedata.p)

#define a function to generate estimated rate data from model and timedata
def _calc_p(model, timedata, lam):
	'''
	Calculates the reactive continuum of rates (or E, for DAEM) for a given
	``rp.TimeData`` and ``rp.Model`` instance.

	Parameters
	----------
	model : rp.Model
		``rp.Model`` instance containing the A matrix to use for calculation.

	timedata : rp.TimeData
		``rp.Timedata`` instance containing the fraction remaining with time 
		array to use for the calculation.

	lam : scalar
		Tikhonov regularization weighting factor, `lambda`.

	Returns
	-------
	p : np.ndarray
		Array of the pdf of the discretized distribution of rates (or E, for 
		DAEM).

	resid : float
		Residual RMSE between true and modeled time data.

	rgh : float
		Roughness RMSE from Tikhonov Regularization.

	References
	----------
	[1] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.

	[2] P.C. Hansen (1994) Regularization tools: A Matlab package for analysis
		and solution of discrete ill-posed problems. *Numerical Algorithms*, 
		**6**, 1-35.
	'''

	#extract nt and nk (or nE for daem)
	nt, nk = np.shape(model.A)

	#calculate the regularization matrix
	R = _calc_R(nk)

	#concatenate A+R and g+zeros
	A_reg = np.concatenate(
		(model.A, R*lam))

	g_reg = np.concatenate(
		(timedata.g, np.zeros(nk + 1)))

	#calculate inverse results and estimated g
	p, _ = nnls(A_reg, g_reg)
	ghat = np.inner(model.A, p)
	rgh = np.inner(R, p)

	#calculate errors
	resid = norm(timedata.g - ghat)/nt**0.5
	rgh = norm(rgh)/nk**0.5

	return p, resid, rgh

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

	[2] P.C. Hansen (1994) Regularization tools: A Matlab package for analysis
		and solution of discrete ill-posed problems. *Numerical Algorithms*, 
		**6**, 1-35.
	'''

	R = np.zeros([n+1, n])

	#ensure pdf = 0 outside of E range specified
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
def _rpo_calc_A(E, log10omega, t, T):
	'''
	Calculates the A matrix for a DAEM model (e.g. a Ramped Pyrox run).

	Parameters
	----------
	E : array-like
		Array of activation energy points to be used in the A matrix, in kJ.
		Length `nE`.

	log10omega : scalar or array-like
		Arrhenius pre-exponential factor, either a constant value, array with
		length `nE`, or a lambda function of E.

	t : array-like
		Array of timepoints to be used in the A matrix, in seconds. Length `nt`.

	T : array-like
		Array of temperature to be used in the A matrix, in Kelvin. Length `nt`.

	Returns
	-------
	A : np.ndarray
		2d array of the Laplace transform for the Daem model. 
		Shape [`nt` x `nE`].

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
	nE = len(E)
	R = 8.314/1000 #kJ/mol/K

	#get arrays in the right format and ensure lengths
	E = assert_len(E, nE) #kJ
	t = assert_len(t, nt) #s
	T = assert_len(T, nt) #K

	#get log10omega into the right format
	if hasattr(log10omega,'__call__'):
		log10omega = log10omega(E)
	
	log10omega = assert_len(log10omega, nE) 
	omega = 10**log10omega #s-1

	#calculate time and E gradients
	dt = np.gradient(t)
	dE = np.gradient(E)

	#pre-allocate A
	A = np.zeros([nt,nE])

	#loop through each timestep
	for i, _ in enumerate(T):

		#generate arrays
		U = T[:i] #Kelvin, [1,i]
		dtau = dt[:i] #seconds, [1,i]

		#generate matrices
		eps_mat = np.outer(E, np.ones(i)) #kJ, [nE,i]
		omega_mat = np.outer(omega, np.ones(i)) #s-1, [nE,i]
		u_mat = np.outer(np.ones(nE), U) #Kelvin, [nE,i]

		#generate A for row i (i.e. for each timepoint) and store in A
		hE_mat = -omega_mat*dtau*np.exp(-eps_mat/(R*u_mat)) #unitless, [nE,i]
		A[i] = np.exp(np.sum(hE_mat, axis = 1))*dE #kJ/mol

	return A

##define function to calculte the A matrix for DAEM models
def _bd_calc_A(k, t):
	'''
	Calculates the A matrix for a BioDecay model (e.g. an IsoCaRB run).

	Parameters
	----------
	k : array-like
		Array of first-order rate constants to be used in the A matrix, in s-1.
		Length `nk`.

	t : array-like
		Array of timepoints to be used in the A matrix, in seconds. Length `nt`.

	Returns
	-------
	A : np.ndarray
		2d array of the Laplace transform for the first-order decay model. 
		Shape [`nt` x `nk`].

	References
	----------
	[1] B.P. Boudreau and B.R. Ruddick (1991) On a reactive continuum
		representation of organic matter diagenesis. *Am. J. Sci.*, **291**,
		507-538.

	[2] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.
	'''

	#set constants
	nt = len(t)
	nk = len(k)

	#get arrays in the right format and ensure lengths
	k = assert_len(k, nk) #s-1
	t = assert_len(t, nt) #s

	#calculate k gradient
	dk = np.gradient(k)

	#pre-allocate A
	A = np.zeros([nt,nk])

	#generate t and k matrices
	tmat = np.outer(t, np.ones(nk))
	kmat = np.outer(np.ones(nt), k)

	#generate and dk matrix
	dkmat = np.outer(np.ones(nt), dk)

	#calculate A matrix
	A = np.exp(-kmat*tmat)*dkmat #s-1

	return A
