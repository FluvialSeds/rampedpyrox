'''
This module contains helper functions for the ratedata classes.
'''

import  numpy as np
import pandas as pd
import warnings

from scipy.optimize import least_squares


#define a function to deconvolve f
def _deconvolve(k, f, nPeaks = 'auto', peak_shape = 'Gaussian', 
	thres = 0.05):
	'''
	Deconvolves f into individual peaks.

	Parameters
	----------
	k : array-like
		Array of k/Ea values considered in the model.

	f : array-like
		Array of a discretized pdf of the distribution of k/Ea values.

	Keyword Arguments
	-----------------
	nPeaks : int or 'auto'
		Tells the program how many peaks to retain after deconvolution.
		Defaults to 'auto'.

	peak_shape : str
		Peak shape to use for deconvolved peaks. Acceptable strings are:
			'Gaussian'
			'(add more later)'
		Defaults to 'Gaussian'.

	thres : float
		Threshold for peak detection cutoff. `thres` is the relative 
		height of the global maximum under which no peaks will be 
		detected. Defaults to 0.05 (i.e. 5% of the highest peak).

	Returns
	-------
	peaks : np.ndarray
		2d array of the k/Ea peaks at each point in k.

	peak_info : np.ndarray
		2d array of the mu, sigma, and height of each peak.

	Raises
	------
	TypeError
		If `nPeaks` is not int or 'auto'.

	ValueError
		If `peak_shape` is not an acceptable string.

	TypeError
		If `thres` is not a float.

	Warnings
	--------
	Warns if ``scipy.optimize.least_squares`` cannot converge on a solution.

	Notes
	-----
	`peak_info` stores the peak information **before** being combined!

	'''

	#assert types
	if not isinstance(nPeaks, int):
		if nPeaks not in ['auto', 'Auto']:
			raise TypeError('nPeaks must be int or "auto"')

	if peak_shape not in ['gaussian','Gaussian']:
		raise ValueError('peak_shape must be one of: Gaussian, (add more)')

	if not isinstance(thres, float):
		raise TypeError('thres must be float')


	#find peak indices and bounds
	ind, lb_ind, ub_ind = _peak_indices(f, nPeaks=nPeaks, thres=thres)

	#calculate initial guess parameters
	n = len(ind)
	mu0 = k[ind]
	sigma0 = 10*np.ones(n) #arbitrarily guess sigma = 10
	height0 = f[ind]

	#pack together for least_squares
	params = np.hstack((mu0,sigma0,height0))

	#calculate bounds
	lb_mu = k[lb_ind]; ub_mu = k[ub_ind]
	lb_sig = np.zeros(n); ub_sig = np.ones(n)*np.max(k)/2.
	lb_height = np.zeros(n); ub_height = np.ones(n)*np.max(f)

	lb = np.hstack((lb_mu,lb_sig,lb_height))
	ub = np.hstack((ub_mu,ub_sig,ub_height))
	bounds = (lb,ub)

	#run least-squares fit
	res = least_squares(_f_phi_diff, params,
		args=(k, f, peak_shape),
		bounds=bounds,
		method='trf')

	#ensure success
	if not res.success:
		warnings.warn('least_squares could not converge on a successful fit!')

	#extract best-fit parameters
	mu = res.x[:n]
	sigma = res.x[n:2*n]
	height = res.x[2*n:]

	#calculate peak info
	peak_info = np.column_stack((mu, sigma, height))

	#calculate peak arrays
	_, peaks = _calc_phi(k, mu, sigma, height, peak_shape)

	return peaks, peak_info

#define function to generate Gaussian peaks
def _gaussian(x, mu, sigma):
	'''
	Calculates a Gaussian peak for a given x vector, mu, and sigma.

	Parameters
	----------
	x : np.ndarray
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
		Array of resulting y values of shape [len(x) x len(mu)].

	Raises
	------
	ValueError
		If mu and sigma arrays are not the same length.
		
	ValueError
		If mu and sigma arrays are not int, float, or np.ndarray.
	'''

	#check data types and broadcast if necessary
	if isinstance(mu, (int, float)) and isinstance(sigma, (int, float)):
		#ensure mu and sigma are floats
		mu = float(mu)
		sigma = float(sigma)

	elif isinstance(mu, np.ndarray) and isinstance(sigma, np.ndarray):
		if len(mu) is not len(sigma):
			raise ValueError('mu and sigma arrays must have same length')

		#ensure mu and sigma dtypes are float
		mu = mu.astype(float)
		sigma = sigma.astype(float)

		#broadcast x into matrix
		n = len(mu)
		x = np.outer(x,np.ones(n))

	else:
		raise ValueError('mu and sigma must be float, int, or np.ndarray')

	#calculate scalar to make sum equal to unity
	scalar = (1/np.sqrt(2.*np.pi*sigma**2))

	#calculate Gaussian
	y = scalar*np.exp(-(x-mu)**2/(2.*sigma**2))

	return np.array(y)

#define a function to find the indices of each peak in `k`.
def _peak_indices(f, nPeaks='auto', thres=0.05):
	'''
	Finds the indices and the bounded range of the mu values for peaks in f.

	Parameters
	----------
	f : np.ndarray
		Array of the pdf of the discretized distribution of Ea/k, f.

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
		Array of indices in `f` containing peak `mu` values.

	lb_ind : np.ndarray
		Array of indices in `f` containing the lower bound for each peak
		`mu` value.

	ub_ind : np.ndarray
		Array of indices in `f` containing the upper bound for each peak
		`mu` value.

	Raises
	------
	ValueError
		If `ub_ind` and `lb_ind` arrays are not the same length.
		
	ValueError
		If `nPeaks` is greater than the total number of peaks detected.
		
	ValueError
		If `nPeaks` is not 'auto' or int.

	Notes
	-----
	This method calculates peaks according to changes in curvature in the
	`f` array. Each bounded section with a negative second derivative (i.e.
	concave down) and `f` value above `thres` is considered a unique peak.
	If `nPeaks` is not 'auto', these peaks are sorted according to decreasing
	peak heights and the first `nPeaks` peaks are saved.
	'''

	#convert thres to absolute value
	thres = thres*(np.max(f)-np.min(f))+np.min(f)

	#calculate derivatives
	df = np.gradient(f)
	d2f = np.gradient(df)

	#calculate bounds such that second derivative is <= 0
	lb_ind = np.where(
		(d2f <= 0) &
		(np.hstack([0.,d2f[:-1]]) > 0))[0]

	ub_ind = np.where(
		(d2f > 0) &
		(np.hstack([0.,d2f[:-1]]) <= 0))[0]

	#remove first UB (initial increase), last LB (final decrease), and check len
	ub_ind = ub_ind[1:]
	lb_ind = lb_ind[:-1]
	if len(ub_ind) is not len(lb_ind):
		raise ValueError('UB and LB arrays have different lenghts')

	#find index of minimum d2f within each bounded range
	ind = []
	for i,j in zip(lb_ind,ub_ind):
		ind.append(i+np.argmin(d2f[i:j]))
	#convert ind to ndarray
	ind = np.array(ind)

	#remove peaks below threshold
	ab = np.where(f[ind] >= thres)
	ind = ind[ab]; lb_ind = lb_ind[ab]; ub_ind = ub_ind[ab]

	#retain first nPeaks according to decreasing f[mu]
	if isinstance(nPeaks,int):
		#check if nPeaks is greater than the total amount of peaks
		if len(ind) < nPeaks:
			raise ValueError('nPeaks greater than total detected peaks')

		#sort according to decreasing f, keep first nPeaks, and re-sort
		ind_sorted = np.argsort(f[ind])[::-1]
		i = np.sort(ind_sorted[:nPeaks])

		ind = ind[i]; lb_ind = lb_ind[i]; ub_ind = ub_ind[i]

	elif nPeaks is not 'auto':
		raise ValueError('nPeaks must be "auto" or int')

	return ind, lb_ind, ub_ind

#define function to generate fitted f distribution, phi
def _calc_phi(k, mu, sigma, height, peak_shape):
	'''
	Calculates phi for given parameters and peak shape

	Parameters
	----------
	k : np.ndarray
		Array of k/Ea values, length nk.

	mu : np.ndarray
		Array of peak means, length nPeak.

	sigma : np.ndarray
		Array of peak standard deviations, length nPeak.

	height : np.ndarray
		Array of peak heights (unitless), length nPeak.

	Returns
	-------
	phi : np.ndarray
		Array of the estimated pdf using the inputted peak parameters.

	peaks : np.ndarray
		Array of individual estimated Ea Gaussian peaks. Shape is 
		[len(eps) x len(mu)].
	'''

	if peak_shape == 'Gaussian':
		#generate Gaussian peaks
		y = _gaussian(k, mu, sigma)

	else:
		raise ValueError((
			"Peak shape: %r is not recognized. Peak shape must be:"
			"	Gaussian," % peak_shape))

	#scale peaks to inputted height
	H = np.max(y, axis=0)
	peaks = y*height/H

	#calculate phi
	phi = np.sum(peaks, axis=1)

	return phi, peaks

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
		Array of k/Ea values, length nk.

	f : np.ndarray
		Array of the pdf of the discretized distribution of Ea/k, f.

	Returns
	-------
	diff : np.ndarray
		Array of the difference between `phi` and `f` at each point,
		length nk.

	Raises
	------
	ValueError
		If len(params) is not 3*n, where n is the length of the `mu`, `sigma`,
		and `height` vectors.

	'''

	n = int(len(params)/3)
	if n != len(params)/3:
		raise ValueError('params array must be length 3n (mu, sigma, height)')

	#unpack parameters
	mu = params[:n]
	sigma = params[n:2*n]
	height = params[2*n:]


	#calculate phi
	phi, _ = _calc_phi(k, mu, sigma, height, peak_shape)

	return phi - f





