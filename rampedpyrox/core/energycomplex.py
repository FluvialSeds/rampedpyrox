'''
``Energycomplex`` module for deconvolving a given Ea distribution, phi, into 
individual Gaussian peaks.

* TODO: Update how _peak_indices sorts to select first nPeaks.
'''

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from numpy.linalg import norm
from scipy.optimize import least_squares

__docformat__ = 'restructuredtext en'

## PRIVATE FUNCTIONS ##

#define function to deconvolve phi
def _deconvolve(eps, phi, nPeaks='auto', thres=0.05):
	'''
	Performs Gaussian peak deconvolution.
	Called by ``EnergyComplex.__init__()``.

	Parameters
	----------
	eps : np.ndarray
		Array of Ea values (in kJ/mol), length nE.

	phi : np.ndarray
		Array of the pdf of the discretized distribution of Ea, f(Ea).

	nPeaks : int or str
		Number of Gaussians to use in deconvolution, either an integer or 
		'auto'. Defaults to 'auto'.

	thres : float
		Threshold for peak detection cutoff. `thres` is the relative height of
		the global maximum under which no peaks will be detected. Defaults to
		0.05 (i.e. 5 percent of the highest peak).

	Returns
	-------
	mu : np.ndarray)
		Array of resulting Gaussian peak means (in kJ), length nPeak.

	sigma : np.ndarray
		Array of resulting Gaussian peak standard deviations (in kJ), length
		nPeak.

	height : np.ndarray
		Array of resulting Gaussian peak heights (unitless), length nPeak.

	Notes
	-----
	`mu` values are bounded to be within each unique concave down region of
	`phi`, `sigma` values are bounded to be below 1/2 of the total `eps`
	range, and `height` values are bounded to be non-negative.
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

#define function to generate Gaussian peaks
def _gaussian(x, mu, sigma):
	'''
	Calculates a Gaussian peak for a given x vector, mu, and sigma.
	Called by ``_phi_hat()``.

	Parameters
	----------
	x : np.ndarray
		Array of x values for Gaussian calculation.

	mu : int, float, or np.ndarray
		Gaussian means, either a single scalar for one peak or an array for 
		simultaneously calculating multiple peaks.

	sigma : int, float, or np.ndarray
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

#define a function to find the indices of each peak in `eps`.
def _peak_indices(phi, nPeaks='auto', thres=0.05):
	'''
	Finds the indices and the bounded range of the mu values for peaks in phi.
	Called by ``_deconvolve()``.

	Parameters
	----------
	phi : np.ndarray
		Array of the pdf of the discretized distribution of Ea, f(Ea).

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

#define function to generate fitted f(Ea) distribution, phi_hat
def _phi_hat(eps, mu, sigma, height):
	'''
	Calculates phi hat for given parameters.
	Called by ``_phi_hat_diff()``.
	Called by ``EnergyComplex.__init__()``.

	Parameters
	----------
	eps : np.ndarray
		Array of Ea values (in kJ/mol), length nE.

	mu : np.ndarray)
		Array of resulting Gaussian peak means (in kJ), length nPeak.

	sigma : np.ndarray
		Array of resulting Gaussian peak standard deviations (in kJ), length
		nPeak.

	height : np.ndarray
		Array of resulting Gaussian peak heights (unitless), length nPeak.

	Returns
	-------
	phi_hat : np.ndarray
		Array of the estimated pdf of the discretized distribution of Ea,
		f(Ea), using the inputted Gaussian peaks.

	y_scaled : np.ndarray
		Array of individual estimated Ea Gaussian peaks. Shape is 
		[len(eps) x len(mu)].
	'''

	#generate Gaussian peaks
	y = _gaussian(eps,mu,sigma)

	#scale peaks to inputted height
	H = np.max(y,axis=0)
	y_scaled = y*height/H

	#calculate phi_hat
	phi_hat = np.sum(y_scaled,axis=1)

	return phi_hat, y_scaled

#define function to calculate the difference between true and estimated phi.
def _phi_hat_diff(params, eps, phi):
	'''
	Calculates the difference between phi and phi_hat for scipy least_squares.
	Called by ``_deconvolve()``.

	Parameters
	----------
	params : np.ndarray
		Array of hrizontally stacked parameter values in the order:
		[`mu`, `sigma`, `height`].

	eps : np.ndarray
		Array of Ea values (in kJ/mol), length nE.

	phi : np.ndarray
		Array of the pdf of the discretized distribution of Ea, f(Ea).

	Returns
	-------
	diff : np.ndarray
		Array of the difference between `phi_hat` and `phi` at each point,
		length nE.

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
	phi_hat,_ = _phi_hat(eps, mu, sigma, height)

	return phi_hat - phi

#define a function to calculate relative areas.
def _rel_area(eps, y_scaled):
	'''
	Calculates the relative contribution of each Ea Gaussian peak to `phi_hat`.

	Parameters
	----------
	eps : np.ndarray
		Array of Ea values (in kJ/mol), length nE.

	y_scaled : np.ndarray
		Array of individual estimated Ea Gaussian peaks. Shape is 
		[len(eps) x len(mu)].

	Returns
	-------
	rel_area : np.ndarray
		Array of relative contributions of each peak to `phi_hat`.
	'''

	#extract gradient and mulitply to get areas
	_,nPeaks = np.shape(y_scaled)
	grad_mat = np.outer(np.gradient(eps),np.ones(nPeaks))
	y_scaled_area = y_scaled*grad_mat

	#calculate relative area
	rel_area = np.sum(y_scaled_area,axis=0)/np.sum(y_scaled_area)

	return rel_area


class EnergyComplex(object):
	__doc__='''
	Class for storing Ea distribution and performing Gaussian deconvolution.

	Parameters
	----------
	eps : np.ndarray
		Array of Ea values (in kJ/mol), length nE.

	phi : np.ndarray
		Array of the pdf of the discretized distribution of Ea, f(Ea).

	nPeaks : int or str
		Number of Gaussians to use in deconvolution, either an integer or 
		'auto'. Defaults to 'auto'.

	thres : float
		Threshold for peak detection cutoff. `thres` is the relative height of
		the global maximum under which no peaks will be detected. Defaults to
		0.05 (i.e. 5 percent of the highest peak).

	combine_last : int or None
		Number of peaks to combine at the end of the run (necessary if there
		is not enough isotope resolution at the high temperature range, as is
		often the case with Ramped PyrOx samples). Defaults to None.

	DEa : int, float, or np.ndarray
		DEa values (the difference in Ea between 12C- and 13C-containing
		molecules) for each Gaussian peak (in kJ/mol), either a scalar or 
		vector of length nPeaks. If using nPeaks = 'auto', leave DEa as a 
		scalar to avoid issues with array length. Defaults to 0.0018, the
		best-fit value determined for carbonate standards on the NOSAMS Ramped
		PyrOx instrument [see Hemingway et al. **(in prep)**].

	Raises
	------
	ValueError
		If phi and eps vectors are not the same length.

	ValueError
		If `nPeaks` is greater than the total number of peaks detected (if
		`nPeaks` is not 'auto').
		
	ValueError
		If `nPeaks` is not 'auto' or int.

	ValueError
		If `DEa` is not int, float, or np.ndarray of length nPeaks.

	Warnings
	--------
	Raises warning if ``scipy.optimize.least_squares`` cannot converge on a
	best-fit solution.

	Notes
	-----
	Peaks are selected according to changes in curvature in the
	`phi` array. Each bounded section with a negative second derivative (i.e.
	concave down) and `phi` value above `thres` is considered a unique peak.
	If `nPeaks` is not 'auto', these peaks are sorted according to decreasing
	peak heights and the first `nPeaks` peaks are saved.

	During peak deconvolution, `mu` values are bounded to be within each
	unique concave down region of `phi`, `sigma` values are bounded to be
	below 1/2 of the total `eps` range, and `height` values are bounded to be
	non-negative.

	Peak deconvolution performed here implements the 
	``scipy.optimize.least_squares`` method using the 'Trust Region
	Reflective' algorithm, as this algorithm is able to handle bounded
	parameters much better than the Levenberg-Marquardt algorithm.


	See Also
	--------
	rampedpyrox.LaplaceTransform.calc_EC_inv
		Method to generate `phi` vector inputted into ``rp.EnergyComplex``.

	Examples
	--------
	Running a thermogram through the inverse model and deconvolving using a
	``rp.LaplaceTransform`` isntance (lt) and ``rp.RealData`` instance (rd)::

		phi,resid_err,rgh_err,omega = lt.calc_fE_inv(rd,omega='auto')
		
		ec = rp.EnergyComplex(eps,phi,
			nPeaks='auto',
			thres=0.02,
			combine_last=3,
			DEa=0.0018)
		
	Plotting resulting f(Ea) distribution and each individual peak::

		#load modules
		import matplotlib.pyplot as plt

		#generate axis handle
		fig,ax = plt.subplots(1,1)

		# plot thermogram
		ax = ec.plot(ax=ax)

	Returning peak summary data from ``rp.EnergyComplex`` instance (ec)::

		#print summary
		ec.summary()

	Attributes
	----------
	eps : np.ndarray
		Array of Ea values (in kJ/mol), length nE.

	height : np.ndarray
		Array of resulting Gaussian peak heights (unitless), length nPeak.

	mu : np.ndarray)
		Array of resulting Gaussian peak means (in kJ), length nPeak.

	peaks : np.ndarray
		Array of individual estimated Ea Gaussian peaks. Shape is 
		[len(eps) x len(mu)].

	phi : np.ndarray
		Array of the pdf of the discretized distribution of Ea, f(Ea).

	phi_hat : np.ndarray
		Array of the estimated pdf of the discretized distribution of Ea,
		f(Ea), using the inputted Gaussian peaks.

	phi_rmse : float
		RMSE value of the estimated `phi_hat` array to `phi` -- that is, the
		Gaussian peak deconvolution RMSE.

	rel_area : np.ndarray
		Array of relative contributions of each peak to `phi_hat`.

	sigma : np.ndarray
		Array of resulting Gaussian peak standard deviations (in kJ), length
		nPeak.

	References
	----------
	\B. de Caprariis et al. (2012) Double-Gaussian distributed activation
	energy model for coal devolatilization. *Energy & Fuels*, **26**,
	6153-6159.

	\B. Cramer (2004) Methane generation from coal during open system 
	pyrolysis investigated by isotope specific, Gaussian distributed reaction
	kinetics. *Organic Geochemistry*, **35**, 379-392.

	J.D. Hemingway et al. **(in prep)** Assessing the blank carbon
	contribution, isotope mass balance, and kinetic isotope fractionation of 
	the ramped pyrolysis/oxidation instrument at NOSAMS.
	'''

	def __init__(self, eps, phi, nPeaks='auto', thres=0.05, combine_last=None, DEa=0.0018):

		#assert phi and eps are same length
		if len(phi) != len(eps):
			raise ValueError('phi and eps vectors must have same length')
		nE = len(phi)

		#assert DEa is int, float, or np.ndarray of the right length
		if not isinstance(DEa, (float,int,np.ndarray)):
			raise ValueError('DEa must be float, int, or np.ndarray of length nPeaks')

		#perform deconvolution, including 13C f(Ea)
		mu,sigma,height = _deconvolve(eps, phi, nPeaks=nPeaks, thres=thres)
		phi_hat,y_scaled = _phi_hat(eps, mu, sigma, height)
		phi_hat_13,y_scaled_13 = _phi_hat(eps, mu+DEa, sigma, height)
		phi_rmse = norm(phi-phi_hat)/nE
		rel_area = _rel_area(eps, y_scaled)

		#define public attributes
		self.phi = phi
		self.eps = eps
		self.mu = mu
		self.sigma = sigma
		self.height = height
		self.phi_hat = phi_hat
		self.phi_rmse = phi_rmse
		self.rel_area = rel_area

		#define private attributes
		self._phi_hat_13 = phi_hat_13

		#combine last peaks if necessary
		if combine_last:
			n = len(mu)-combine_last
			combined = np.sum(y_scaled[:,n:],axis=1)
			combined_13 = np.sum(y_scaled_13[:,n:],axis=1)
			self.peaks = np.column_stack((y_scaled[:,:n],combined))
			self._peaks_13 = np.column_stack((y_scaled_13[:,:n],combined_13))
		else:
			self.peaks = y_scaled
			self._peaks_13 = y_scaled_13

	def plot(self, ax=None):
		'''
		Plots the inverse and peak-deconvolved discretized f(Ea).

		Parameters
		----------
		ax : None or matplotlib.axis
			Axis to plot on. If `None`, automatically creates a
			``matplotlip.axis`` instance to return. Defaults to `None`.

		Returns
		-------
		ax : matplotlib.axis
			Updated axis instance with plotted data.
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

	def summary(self):
		'''
		Prints a summary of the ``rp.EnergyComplex`` instance.
		'''

		#make a pd.DataFrame object
		data = np.column_stack((self.mu, self.sigma, self.height, 
			self.rel_area))
		col_name = ['means (kJ)','stdev. (kJ)','height', 'rel. area']
		ind_name = np.arange(1,len(self.mu)+1)
		df = pd.DataFrame(data,columns=col_name,index=ind_name)

		#make strings
		title = self.__class__.__name__ + ' summary table:'
		line = '==========================================================='
		pi = 'Peak information for each deconvolved peak:'
		note = 'NOTE: Combined peaks are reported separately in this table!'
		RMSE = 'Deconvolution RMSE = %.2f x 10^6' %(self.phi_rmse*1e6)

		print(title + '\n\n' + line + '\n' + pi + '\n\n' + note + '\n')
		print(df)
		print('\n' + line + '\n\n' + RMSE + '\n\n' + line)


if __name__ == '__main__':

	import rampedpyrox.core.api as rp
