'''
Energycomplex module for deconvolving a given Ea distribution into individual
Gaussian peaks.

* TODO: Make legend more pythonic.
* TODO: Fix how _peak_indices sorts to select first nPeaks.
* TODO: Include references for finding peak indices.
'''

import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.linalg import norm
from scipy.optimize import least_squares


def _deconvolve(eps, phi, nPeaks='auto', thres=0.05):
	'''
	Performs Gaussian peak deconvolution.
	Called by ``EnergyComplex.__init__()``.

	Args:
		eps (np.ndarray): Array of Ea values.

		phi (np.ndarray): Array of the pdf of the distribution of Ea.

		nPeaks (int or str): Number of Gaussians to use in deconvolution,
			either an integer or 'auto'. Defaults to 'auto'.

		thres (float): Threshold for peak detection cutoff. Thres is the relative
			height of the global maximum under which no peaks will be detected.

	Returns:
		mu (np.ndarray): Array of resulting Gaussian peak means.

		sigma (np.ndarray): Array of resulting Gaussian peak standard deviations.

		height (np.ndarray): Array of resulting Gaussian peak heights.

	'''

	#find peak indices and bounds
	ind,lb_ind,ub_ind = _peak_indices(phi,nPeaks=nPeaks,thres=thres)

	#calculate initial guess parameters
	n = len(ind)
	mu0 = eps[ind]
	sigma0 = 10*np.ones(n) #arbitrarily guess sigma = 10kJ/mol
	height0 = phi[ind]

	#pack together for least_squares
	params = np.hstack((mu0,sigma0,height0))

	#calculate bounds
	lb_mu = eps[lb_ind]; ub_mu = eps[ub_ind]
	lb_sig = np.zeros(n); ub_sig = np.ones(n)*np.max(eps)/2.
	lb_height = np.zeros(n); ub_height = np.ones(n)*np.max(phi)

	lb = np.hstack((lb_mu,lb_sig,lb_height))
	ub = np.hstack((ub_mu,ub_sig,ub_height))
	bounds = (lb,ub)

	#run least-squares fit
	res = least_squares(_phi_hat_diff,params,
		args=(eps,phi),
		bounds=bounds,
		method='trf')

	#ensure success
	if not res.success:
		warnings.warn('least_squares could not converge on a successful fit')

	#extract best-fit parameters
	mu = res.x[:n]
	sigma = res.x[n:2*n]
	height = res.x[2*n:]

	return mu, sigma, height

def _gaussian(x, mu, sigma):
	'''
	Calculates a Gaussian peak for a given x vector, mu, and sigma.
	Called by ``_phi_hat()``.

	Args:
		x (np.ndarray): Array of x values for Gaussian calculation.

		mu (int, float, or np.ndarray): Gaussian means, either a single scalar
			for one peak or an array for simultaneously calculating multiple
			peaks.

		sigma (int, float, or np.ndarray): Gaussian standard deviations, either
			a single scalar for one peak or an array for simultaneously
			calculating multiple peaks.

	Returns:
		y (np.ndarray): Array of resulting y values of shape [len(x) x len(mu)].

	Raises:
		ValueError: If mu and sigma arrays are not the same length.
		
		ValueError: If mu and sigma arrays are not int, float, or np.ndarray.
	'''

	#check data types and broadcast if necessary
	if isinstance(mu,(int,float)) and isinstance(sigma,(int,float)):
		#ensure mu and sigma are floats
		mu = float(mu)
		sigma = float(sigma)

	elif isinstance(mu,np.ndarray) and isinstance(sigma,np.ndarray):
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

	return y

def _peak_indices(phi, nPeaks='auto', thres=0.05):
	'''
	Finds the indices and the bounded range of the mu values for peaks in phi.
	Called by ``_deconvolve()``.

	Args:
		phi (np.ndarray): Array of the pdf of the distribution of Ea.

		nPeaks (int or str): Number of Gaussians to use in deconvolution,
			either an integer or 'auto'. Defaults to 'auto'.

		thres (float): Threshold for peak detection cutoff. Thres is the relative
			height of the global maximum under which no peaks will be detected.

	Returns:
		ind (np.ndarray): Array of indices in phi containing peak mu values.

		lb_ind(np.ndarray): Array of indices in phi containing the lower bound
			for each peak mu value.

		ub_ind(np.ndarray): Array of indices in phi containing the upper bound
			for each peak mu value.

	Raises:
		ValueError: If ub_ind and lb_ind arrays are not the same length.
		
		ValueError: If ``nPeaks`` is greater than the total number of peaks detected.
		
		ValueError: If ``nPeaks`` is not 'auto' or int.

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

	#retain first nPeaks according to increasing d2phi
	# if isinstance(nPeaks,int):
	# 	#check if nPeaks is greater than the total amount of peaks
	# 	if len(ind) < nPeaks:
	# 		raise ValueError('nPeaks greater than total detected peaks')

	# 	#sort according to increasing d2phi, keep first nPeaks, and re-sort
	# 	i = np.argsort(d2phi[ind])[:nPeaks]
	# 	i = np.sort(i)

	# 	ind = ind[i]; lb_ind = lb_ind[i]; ub_ind = ub_ind[i]

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

def _phi_hat(eps, mu, sigma, height):
	'''
	Calculates phi hat for given parameters.
	Called by ``_phi_hat_diff()``.
	Called by ``EnergyComplex.__init__()``.

	Args:
		eps (np.ndarray): Array of Ea values.

		mu (np.ndarray): Array of Gaussian peak means.

		sigma (np.ndarray): Array of Gaussian peak standard deviations.

		height (np.ndarray): Array of Gaussian peak heights.

	Returns:
		phi_hat (np.ndarray): Array of estimated Ea distribution.

		y_scaled (np.ndarray): Array of individual estimated Ea Gaussian peaks.
			Shape is [len(eps) x len(mu)].
	'''

	#generate Gaussian peaks
	y = _gaussian(eps,mu,sigma)

	#scale peaks to inputted height
	H = np.max(y,axis=0)
	y_scaled = y*height/H

	#calculate phi_hat
	phi_hat = np.sum(y_scaled,axis=1)

	return phi_hat, y_scaled

def _phi_hat_diff(params, eps, phi):
	'''
	Calculates the difference between phi and phi_hat for scipy least_squares.
	Called by ``_deconvolve()``.

	Args:
		params (np.ndarray): Array of hrizontally stacked parameter values with
			shape [mus, sigmas, heights].

		eps (np.ndarray): Array of Ea values.

		phi (np.ndarray): Array of the pdf of the distribution of Ea.

	Returns:
		diff (np.ndarray): Array of the difference between phi_hat and phi
			at each point.

	Raises:
		ValueError: If len(params) is not 3*n, where n is the length of the
			mu, sigma, and height vectors.

	'''

	n = int(len(params)/3)
	if n != len(params)/3:
		raise ValueError('params array must be length 3n (mu, sigma, height)')

	#unpack parameters
	mu = params[:n]
	sigma = params[n:2*n]
	height = params[2*n:]


	#calculate phi_hat
	phi_hat,_ = _phi_hat(eps, mu, sigma, height)

	return phi_hat - phi


class EnergyComplex(object):
	'''
	Class for storing Ea distribution and calculating peak deconvolution.

	Args:
		eps (np.ndarray): Array of Ea values.

		phi (np.ndarray): Array of the pdf of the distribution of Ea.

		nPeaks (int or str): Number of Gaussians to use in deconvolution,
			either an integer or 'auto'. Defaults to 'auto'.

		thres (float): Threshold for peak detection cutoff. Thres is the relative
			height of the global maximum under which no peaks will be detected.

		combine_last (int or None): Number of peaks to combine at the end of
			the run (necessary if there is not enough isotope resolution at the
			high temperature range, as is often the case with real data).
			Defaults to None.

	Returns:
		ec (rp.EnergyComplex): ``EnergyComplex`` object.

	Raises:
		ValueError: If phi and eps vectors are not the same length.

		ValueError: If ``nPeaks`` is greater than the total number of peaks detected.
		
		ValueError: If ``nPeaks`` is not 'auto' or int.

	Examples:
		Running a thermogram through the inverse model and deconvolving::

			#assuming a LaplaceTransform object lt and RealData object rd
			phi,resid_err,rgh_err,omega = lt.calc_fE_inv(rd,omega='auto')
			ec = rp.EnergyComplex(eps,phi,nPeaks='auto',combine_last=None)
			ax = ec.plot()

	References:

	Notes:
		All results are bounded to be non-negative, and mu values are bounded to be
		within the concave down regions of the Ea distribution. This class 
		implements the scipy.optimize.least_squares method using the 'Trust Region
		Reflective' algorithm, as this algorithm is able to handle bounded
		parameters much better than the Levenberg-Marquardt algorithm.
	'''

	def __init__(self, eps, phi, nPeaks='auto', thres=0.05, combine_last=None):

		#assert phi and eps are same length
		if len(phi) != len(eps):
			raise ValueError('phi and eps vectors must have same length')
		nE = len(phi)

		#perform deconvolution
		mu,sigma,height = _deconvolve(eps, phi, nPeaks=nPeaks, thres=thres)
		phi_hat,y_scaled = _phi_hat(eps, mu, sigma, height)
		phi_err = norm(phi-phi_hat)/nE

		#define public parameters
		self.phi = phi
		self.eps = eps
		self.mu = mu
		self.sigma = sigma
		self.height = height
		self.phi_hat = phi_hat
		self.phi_err = phi_err

		#combine last peaks if necessary
		if combine_last:
			n = len(mu)-combine_last
			combined = np.sum(y_scaled[:,n:],axis=1)
			self.peaks = np.column_stack((y_scaled[:,:n],combined))
		else:
			self.peaks = y_scaled

	def plot(self, ax=None):
		'''
		Plots the inverse and peak-deconvolved EC.

		Args:
			ax (None or matplotlib.axis): Axis to plot on. If None, 
			creates an axis object to return. Defaults to None.

		Returns:
			ax (matplotlib.axis): Updated axis with plotted data.

		Examples:
			Basic implementation::

				#assuming EnergyComplex object ec
				ax = ec.plot()
		'''

		if ax is None:
			_,ax = plt.subplots(1,1,figsize=(8,6))

		#plot phi in black
		ax.plot(self.eps,self.phi,
			color='k',
			linewidth=2,
			label=r'Inversion Result $(\phi)$')

		#plot phi_hat in red
		ax.plot(self.eps,self.phi_hat,
			color='r',
			linewidth=2,
			label=r'Peak-fitted estimate $(\hat{\phi})$')

		#plot individual peaks in dashes
		ax.plot(self.eps,self.peaks,
			'--k',
			linewidth=1,
			label=r'Individual fitted Gaussians (n=%.0f)' %len(self.mu))

		#remove duplicate legend entries
		handles, labels = ax.get_legend_handles_labels()
		handle_list, label_list = [], []
		for handle, label in zip(handles, labels):
			if label not in label_list:
				handle_list.append(handle)
				label_list.append(label)
		
		ax.legend(handle_list,label_list,loc='best')

		#label axes
		ax.set_xlabel('Activation Energy (kJ)')
		ax.set_ylabel('f(Ea) (unitless)')

		return ax

	def summary():
		'''
		Prints a summary of the EnergyComplex object.
		'''
