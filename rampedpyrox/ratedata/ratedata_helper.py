'''
This module contains helper functions for the ratedata classes.
'''

import  numpy as np
import pandas as pd
import warnings

from scipy.optimize import least_squares

#import container classes
from rampedpyrox.core.array_classes import(
	rparray
	)

#define a function to deconvolve phi
def _deconvolve(k, phi, nPeaks = 'auto', 
	peak_shape = 'Gaussian', thres = 0.05):
	'''
	Deconvolves phi into individual peaks.

	Parameters
	----------

	Keyword Arguments
	-----------------

	Returns
	-------

	Raises
	------

	'''

	#find peak indices and bounds
	ind, lb_ind, ub_ind = _peak_indices(phi, nPeaks=nPeaks, thres=thres)

	#calculate initial guess parameters
	n = len(ind)
	mu0 = k[ind]
	sigma0 = 10*np.ones(n) #arbitrarily guess sigma = 10
	height0 = phi[ind]

	#pack together for least_squares
	params = np.hstack((mu0,sigma0,height0))

	#calculate bounds
	lb_mu = k[lb_ind]; ub_mu = k[ub_ind]
	lb_sig = np.zeros(n); ub_sig = np.ones(n)*np.max(k)/2.
	lb_height = np.zeros(n); ub_height = np.ones(n)*np.max(phi)

	lb = np.hstack((lb_mu,lb_sig,lb_height))
	ub = np.hstack((ub_mu,ub_sig,ub_height))
	bounds = (lb,ub)

	#run least-squares fit
	res = least_squares(_phi_hat_diff, params,
		args=(k, phi, peak_shape),
		bounds=bounds,
		method='trf')

	#ensure success
	if not res.success:
		warnings.warn('least_squares could not converge on a successful fit')

	#extract best-fit parameters
	mu = res.x[:n]
	sigma = res.x[n:2*n]
	height = res.x[2*n:]

	#calculate peak arrays
	_, peaks = _phi_hat(k, mu, sigma, height, peak_shape)
	peaks = rparray(peaks, len(k))

	#combine peak_info into pandas dataframe
	peak_info = np.column_stack((mu, sigma, height))
	peak_info = pd.DataFrame(peak_info, 
		columns = ['mu', 'sigma', 'height'],
		index = np.arange(1,n + 1))

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
	y : rp.rparray
		Array of resulting y values of shape [len(x) x len(mu)].

	Raises
	------
	ValueError
		If mu and sigma arrays are not the same length.
		
	ValueError
		If mu and sigma arrays are not int, float, or np.ndarray.
	'''

	#check data types and broadcast if necessary
	if isinstance(mu,(int,float)) and isinstance(sigma,(int,float)):
		#ensure mu and sigma are floats
		mu = rparray(mu, 1)
		sigma = rparray(sigma, 1)

	else:
		n = len(mu)
		mu = rparray(mu, n)
		sigma = rparray(sigma, n)

		x = np.outer(x, np.ones(n))

	#calculate scalar to make sum equal to unity
	scalar = (1/np.sqrt(2.*np.pi*sigma**2))

	#calculate Gaussian
	y = scalar*np.exp(-(x-mu)**2/(2.*sigma**2))

	return y

#define a function to find the indices of each peak in `k`.
def _peak_indices(phi, nPeaks='auto', thres=0.05):
	'''
	Finds the indices and the bounded range of the mu values for peaks in phi.

	Parameters
	----------
	phi : np.ndarray
		Array of the pdf of the discretized distribution of Ea/k, phi.

	nPeaks : int or str
		Number of Gaussians to use in deconvolution, either an integer or 
		'auto'. Defaults to 'auto'.

	thres : float
		Threshold for peak detection cutoff. `thres` is the relative height of
		the global maximum under which no peaks will be detected. Defaults to
		0.05 (i.e. 5% of the highest peak).

	Returns
	-------
	ind : np.ndarray)
		Array of indices in `phi` containing peak `mu` values.

	lb_ind : np.ndarray
		Array of indices in `phi` containing the lower bound for each peak
		`mu` value.

	ub_ind : np.ndarray
		Array of indices in `phi` containing the upper bound for each peak
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
	`phi` array. Each bounded section with a negative second derivative (i.e.
	concave down) and `phi` value above `thres` is considered a unique peak.
	If `nPeaks` is not 'auto', these peaks are sorted according to decreasing
	peak heights and the first `nPeaks` peaks are saved.
	'''

	#convert thres to absolute value
	thres = thres*(np.max(phi)-np.min(phi))+np.min(phi)

	#calculate derivatives
	dphi = np.gradient(phi)
	d2phi = np.gradient(dphi)

	#calculate bounds such that second derivative is <= 0
	lb_ind = np.where(
		(d2phi <= 0) &
		(np.hstack([0.,d2phi[:-1]]) > 0))[0]

	ub_ind = np.where(
		(d2phi > 0) &
		(np.hstack([0.,d2phi[:-1]]) <= 0))[0]

	#remove first UB (initial increase), last LB (final decrease), and check len
	ub_ind = ub_ind[1:]
	lb_ind = lb_ind[:-1]
	if len(ub_ind) is not len(lb_ind):
		raise ValueError('UB and LB arrays have different lenghts')

	#find index of minimum d2phi within each bounded range
	ind = []
	for i,j in zip(lb_ind,ub_ind):
		ind.append(i+np.argmin(d2phi[i:j]))
	#convert ind to ndarray
	ind = np.array(ind)

	#remove peaks below threshold
	ab = np.where(phi[ind] >= thres)
	ind = ind[ab]; lb_ind = lb_ind[ab]; ub_ind = ub_ind[ab]

	#retain first nPeaks according to decreasing phi[mu]
	if isinstance(nPeaks,int):
		#check if nPeaks is greater than the total amount of peaks
		if len(ind) < nPeaks:
			raise ValueError('nPeaks greater than total detected peaks')

		#sort according to decreasing phi, keep first nPeaks, and re-sort
		ind_sorted = np.argsort(phi[ind])[::-1]
		i = np.sort(ind_sorted[:nPeaks])

		ind = ind[i]; lb_ind = lb_ind[i]; ub_ind = ub_ind[i]

	elif nPeaks is not 'auto':
		raise ValueError('nPeaks must be "auto" or int')

	return ind, lb_ind, ub_ind

#define function to generate fitted phi distribution, phi_hat
def _phi_hat(k, mu, sigma, height, peak_shape):
	'''
	Calculates phi hat for given parameters and peak shape

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
	phi_hat : rp.rparray
		Array of the estimated pdf using the inputted peak parameters.

	y_scaled : np.ndarray
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
	y_scaled = y*height/H

	#calculate phi_hat
	phi_hat = np.sum(y_scaled, axis=1)

	return phi_hat, y_scaled

#define function to calculate the difference between true and estimated phi.
def _phi_hat_diff(params, k, phi, peak_shape):
	'''
	Calculates the difference between phi and phi_hat for scipy least_squares.

	Parameters
	----------
	params : np.ndarray
		Array of hrizontally stacked parameter values in the order:
		[`mu`, `sigma`, `height`].

	k : np.ndarray
		Array of k/Ea values, length nk.

	phi : np.ndarray
		Array of the pdf of the discretized distribution of Ea/k, phi.

	Returns
	-------
	diff : np.ndarray
		Array of the difference between `phi_hat` and `phi` at each point,
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


	#calculate phi_hat
	phi_hat, _ = _phi_hat(k, mu, sigma, height, peak_shape)

	return phi_hat - phi





















