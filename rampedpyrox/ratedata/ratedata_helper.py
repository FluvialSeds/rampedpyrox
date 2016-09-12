'''
This module contains helper functions for the ratedata classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_calc_phi', '_deconvolve', '_f_phi_diff', '_gaussian', 
			'_peak_indicies']

import  numpy as np
import pandas as pd
import warnings

from scipy.optimize import least_squares

#import exceptions
from ..core.exceptions import(
	ArrayError,
	FitError,
	ScalarError,
	StringError,
	)

#import helper functions
from ..core.core_functions import(
	assert_len,
	)

#define function to generate fitted f distribution, phi
def _calc_phi(k, mu, sigma, height, peak_shape):
	'''
	Calculates phi for given parameters and peak shape

	Parameters
	----------
	k : array-like
		Array of k/Ea values, length `nk`.

	mu : int, float, or array-like
		Peak means (kJ/mol), either a single scalar for one peak or an 
		array for simultaneously calculating multiple peaks. If array, 
		length `nPeak`.

	sigma : int, float, or array-like
		Peak standard deviations (kJ/mol), either a single scalar for one 
		peak or an array for simultaneously calculating multiple peaks. If 
		array, length `nPeak`.

	height : int, float, or array-like
		Peak heights (unitless), either a single scalar for one peak or an
		array for simultaneously calculating multiple peaks. If array,
		length `nPeak`.

	Returns
	-------
	phi : np.ndarray
		Array of the estimated pdf using the inputted peak parameters.
		Length `nk`.

	peaks : np.ndarray
		Array of individual estimated Ea Gaussian peaks. Shape 
		[`nk` x `nPeak`].
	
	Raises
	------
	StringError
		If `peak_shape` is not an acceptable string.
	'''

	if peak_shape in ['Gaussian', 'gaussian']:
		#generate Gaussian peaks
		y = _gaussian(k, mu, sigma)

	else:
		raise StringError(
			'Peak shape: %r is not recognized. Peak shape must be:'
			'	Gaussian,' % peak_shape)

	#check that height is the same shape as mu and sigma
	if isinstance(mu, (int, float)):
		#ensure float
		height = float(height)

	else:
		n = len(mu)

		#assert height is array with same shape as mu
		height = assert_len(height, n)


	#scale peaks to inputted height
	H = np.max(y, axis=0)
	peaks = y*height/H

	#calculate phi
	phi = np.sum(peaks, axis=1)

	return phi, peaks

#define a function to deconvolve f
def _deconvolve(
		k, 
		f, 
		nPeaks = 'auto', 
		peak_shape = 'Gaussian', 
		thres = 0.05):
	'''
	Deconvolves f into individual peaks.

	Parameters
	----------
	k : array-like
		Array of k/Ea values considered in the model. Length `nk`.

	f : array-like
		Array of a discretized pdf of the distribution of k/Ea values.
		Length `nk`.

	nPeaks : int or 'auto'
		Tells the program how many peaks to retain after deconvolution.
		Defaults to 'auto'.

	peak_shape : str
		Peak shape to use for deconvolved peaks. Acceptable strings are:

			'Gaussian'
		
		Defaults to 'Gaussian'.

	thres : float
		Threshold for peak detection cutoff. `thres` is the relative 
		height of the global maximum under which no peaks will be 
		detected. Defaults to 0.05 (i.e. 5% of the highest peak).

	Returns
	-------
	peaks : np.ndarray
		2d array of the k/Ea peaks at each point in `k`. Shape 
		[`nk` x `nPeak`].

	peak_info : np.ndarray
		2d array containing the inverse-modeled peak isotope summary info:

			mu (kJ/mol), \n
			sigma (kJ/mol), \n
			height (unitless), \n

	Warnings
	--------
	UserWarning
		If ``scipy.optimize.least_squares`` cannot converge on a solution.

	Notes
	-----
	`peak_info` stores the peak information **before** being combined!
	After finding best-fit peak parameters, heights are scaled uniformly such
	that the sum of all peaks integreates to unity (*i.e.* ensures that phi is
	a pdf).

	'''

	#find peak indices and bounds
	ind, lb_ind, ub_ind = _peak_indices(
		f, 
		nPeaks = nPeaks, 
		thres = thres)

	#calculate initial guess parameters
	n = len(ind)
	mu0 = k[ind]
	sigma0 = 10*np.ones(n) #arbitrarily guess sigma = 10
	height0 = f[ind]

	#pack together for least_squares
	params = np.hstack(
		(mu0, sigma0, height0))

	#calculate bounds
	lb_mu = k[lb_ind]
	ub_mu = k[ub_ind]
	
	lb_sig = np.zeros(n)
	ub_sig = np.ones(n)*np.max(k)/2.
	
	lb_height = np.zeros(n)
	ub_height = np.ones(n)*np.max(f)

	lb = np.hstack(
		(lb_mu,lb_sig,lb_height))

	ub = np.hstack(
		(ub_mu,ub_sig,ub_height))

	bounds = (lb,ub)

	#run least-squares fit
	res = least_squares(
		_f_phi_diff, 
		params,
		args=(k, f, peak_shape),
		bounds=bounds,
		method='trf')

	#ensure success
	if not res.success:
		warnings.warn(
			'least_squares could not converge on a successful fit!',
			UserWarning)

	#extract best-fit parameters
	mu = res.x[:n]
	sigma = res.x[n:2*n]
	height = res.x[2*n:]

	#calculate peak arrays
	phi, peaks = _calc_phi(k, mu, sigma, height, peak_shape)

	#scale peak heights to ensure peaks integrates to one
	a = np.sum(phi*np.gradient(k))
	peaks = peaks/a
	height = height/a

	#combine peak info
	peak_info = np.column_stack(
		(mu, sigma, height))

	return peaks, peak_info

#define function to calculate the difference between true and estimated f(Ea).
def _f_phi_diff(params, k, f, peak_shape):
	'''
	Calculates the difference between f and phi for scipy least_squares.

	Parameters
	----------
	params : np.ndarray
		Array of hrizontally stacked parameter values in the order:
		[`mu`, `sigma`, `height`].

	k : np.ndarray
		Array of k/Ea values, length `nk`.

	f : np.ndarray
		Array of the pdf of the discretized distribution of Ea/k, f.
		Length `nk`.

	Returns
	-------
	diff : np.ndarray
		Array of the difference between `phi` and `f` at each point,
		length `nk`.

	Raises
	------
	ArrayError
		If len(params) is not ``3*n``, where `n` is the length of the `mu`, 
		`sigma`, and `height` vectors.

	'''

	if len(params) % 3 != 0:
		raise ArrayError(
			'params array must be length 3n (mu, sigma, height)')

	n = int(len(params)/3)

	#unpack parameters
	mu = params[:n]
	sigma = params[n:2*n]
	height = params[2*n:]

	#calculate phi
	phi, _ = _calc_phi(k, mu, sigma, height, peak_shape)

	return phi - f

#define function to generate Gaussian peaks
def _gaussian(x, mu, sigma):
	'''
	Calculates a Gaussian peak for a given x vector, mu, and sigma.

	Parameters
	----------
	x : array-like
		Array of x values for Gaussian calculation.

	mu : int, float, or array-like
		Gaussian means, either a single scalar for one peak or an array for 
		simultaneously calculating multiple peaks.

	sigma : int, float, or array-like
		Gaussian standard deviations, either a single scalar for one peak or 
		an array for simultaneously calculating multiple peaks.

	Returns
	-------
	y : np.ndarray
		Array of resulting y values of shape [`len(x)` x `len(mu)`].
	'''

	#check data types and broadcast if necessary
	x = assert_len(x, len(x))

	if isinstance(mu, (int, float)) and isinstance(sigma, (int, float)):
		#ensure mu and sigma are floats
		mu = float(mu)
		sigma = float(sigma)

	else:
		n = len(mu)

		#assert mu and sigma are array-like and the same shape
		mu = assert_len(mu, n)
		sigma = assert_len(sigma, n)

		#broadcast x into matrix
		x = np.outer(x, np.ones(n))

	#calculate scalar to make sum equal to unity
	scalar = (1/np.sqrt(2.*np.pi*sigma**2))

	#calculate Gaussian
	y = scalar*np.exp(-(x-mu)**2/(2.*sigma**2))

	return y

#define a function to find the indices of each peak in `k`.
def _peak_indices(f, nPeaks = 'auto', thres = 0.05):
	'''
	Finds the indices and the bounded range of the mu values for peaks in f.

	Parameters
	----------
	f : np.ndarray
		Array of the pdf of the discretized distribution of Ea/k, f.
		Length `nk`.

	nPeaks : int or str
		Number of peaks to use in deconvolution, either an integer or 
		'auto'. Defaults to 'auto'.

	thres : float
		Threshold for peak detection cutoff. `thres` is the relative height of
		the global maximum under which no peaks will be detected. Defaults to
		0.05 (i.e. 5% of the highest peak).

	Returns
	-------
	ind : np.ndarray
		Array of indices in `f` containing peak `mu` values. Length `nPeak`.

	lb_ind : np.ndarray
		Array of indices in `f` containing the lower bound for each peak
		`mu` value. Length `nPeak`.

	ub_ind : np.ndarray
		Array of indices in `f` containing the upper bound for each peak
		`mu` value. Length `nPeak`.

	Raises
	------
	ArrayError
		If `ub_ind` and `lb_ind` arrays are not the same length.
		
	FitError
		If `nPeaks` is greater than the total number of peaks detected.

	FitError
		If no peaks are detected.
		
	ScalarError
		If `nPeaks` is not 'auto' or int.

	ScalarError
		If `nPeaks` is not int or 'auto'.

	ScalarError
		If `thres` is not a float.

	ScalarError
		If `thres` is not between (0, 1).

	Notes
	-----
	This method calculates peaks according to changes in curvature in the
	`f` array. Each bounded section with a negative second derivative (i.e.
	concave down) and `f` value above `thres` is considered a unique peak.
	If `nPeaks` is not 'auto', these peaks are sorted according to decreasing
	peak heights and the first `nPeaks` peaks are saved.
	'''

	#assert types and strings
	if not isinstance(nPeaks, int):
		if nPeaks not in ['auto', 'Auto']:
			raise ScalarError(
				'nPeaks must be int or "auto"')

	if not isinstance(thres, float):
		raise ScalarError(
			'thres must be float')
	
	elif thres > 1 or thres < 0:
		raise ScalarError(
			'thres must be between 0 and 1 (fractional height)')

	#convert thres to absolute value
	thres = thres*(np.max(f) - np.min(f)) + np.min(f)

	#calculate derivatives
	df = np.gradient(f)
	d2f = np.gradient(df)

	#calculate bounds such that second derivative is <= 0
	lb_ind = np.where(
		(d2f <= 0) &
		(np.hstack([0., d2f[:-1]]) > 0))[0]

	ub_ind = np.where(
		(d2f > 0) &
		(np.hstack([0., d2f[:-1]]) <= 0))[0]

	#first point gets picked up as an upper bound. Remove and check len.
	ub_ind = ub_ind[1:]
	
	if len(ub_ind) != len(lb_ind):
		#final point gets picked up as a lower bound. Remove.
		lb_ind = lb_ind[:-1]

	#if still not the same length, raise error
	if len(ub_ind) != len(lb_ind):
		raise ArrayError(
			'UB and LB arrays have different lengths')
	elif len(ub_ind) == 0:
		raise FitError(
			'No peaks detected!')

	#find index of where df is closest to zero within each bounded range
	ind = np.zeros(len(ub_ind), dtype = int)

	for i, (a, b) in enumerate(zip(lb_ind, ub_ind)):
		# ind[i] = a + np.argmin(d2f[a:b])
		ind[i] = a + np.argmin(np.abs(df[a:b]))

	#remove peaks below threshold
	ab = np.where(f[ind] >= thres)

	ind = ind[ab]
	lb_ind = lb_ind[ab]
	ub_ind = ub_ind[ab]

	#retain first nPeaks according to decreasing f[mu]
	if isinstance(nPeaks,int):
		#check if nPeaks is greater than the total amount of peaks
		if len(ind) < nPeaks:
			raise FitError(
				'nPeaks greater than total detected peaks')

		#sort according to decreasing f, keep first nPeaks, and re-sort
		ind_sorted = np.argsort(f[ind])[::-1]
		
		i = np.sort(ind_sorted[:nPeaks])

		ind = ind[i]
		lb_ind = lb_ind[i]
		ub_ind = ub_ind[i]

	elif nPeaks is not 'auto':
		raise ScalarError(
			'nPeaks must be "auto" or int')

	return ind, lb_ind, ub_ind
