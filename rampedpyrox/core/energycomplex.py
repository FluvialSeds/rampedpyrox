#TODO: Make legend more pythonic.
#TODO: Fix peak fitting algorithm to include shoulders

import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.linalg import norm
from scipy.optimize import least_squares


def _deconvolve(eps, phi, nPeaks = 'auto', thres=0.05):
	'''
	Deconvolves f(Ea) into Gaussian peaks.
	'''

	#find peak indices
	if nPeaks is 'auto':
		ind_sorted = _peak_indices(phi, thres=thres, min_dist=1)
	elif isinstance(nPeaks,int):
		all_ind = _peak_indices(phi, thres=thres, min_dist=1)

		#check if nPeaks is greater than the total amount of peaks
		if len(all_ind) < nPeaks:
			raise ValueError('nPeaks > total detected at current threshold')
		
		ind_sorted = all_ind[:nPeaks]
	else:
		raise ValueError('nPeaks must be "auto" or int')

	#re-sort indices by increasing Ea (rather than decreasing phi)
	ind = np.sort(ind_sorted)
	n = len(ind)

	#calculate initial guess parameters
	mu0 = eps[ind]
	sigma0 = 10*np.ones(n) #arbitrarily guess sigma = 10kJ/mol
	height0 = phi[ind]

	#pack together for least_squares and create bounds
	params = np.hstack((mu0,sigma0,height0))
	lb = np.zeros(3*n)
	ub_mu = np.max(eps)*np.ones(n)
	ub_sig = ub_mu/2
	ub_height = np.max(phi)*np.ones(n)
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

def _peak_indices(phi, thres=0.05, min_dist=1):
	'''
	Finds the indices of the peaks in phi (modified from peakutils.indexes).
	'''

	#convert relative threshold to absolute value
	thres = thres*(np.max(phi)-np.min(phi))+np.min(phi)

	#ensure min_dist is an integer
	min_dist = int(min_dist)

	#find the peaks above threshold by using the first order difference
	#stack with 0 on either side to find when dphi crosses 0 from + to -
	dphi = np.diff(phi)
	ind = np.where((np.hstack([dphi, 0.]) < 0.)
		& (np.hstack([0., dphi]) > 0.)
		& (phi > thres))[0]

	#sort indices from largest to smallest peak
	ind_sorted = ind[np.argsort(phi[ind])][::-1]

	#remove peaks that are too close together
	if len(ind) > 1 and min_dist > 1:
		rem = np.ones(len(phi), dtype=bool)
		rem[ind_sorted] = False

		#starting with highest peak, remove any others within +/- min_dist
		for i in ind_sorted:
			if not rem[i]:
				sl = slice(max(0, i-min_dist), i+min_dist+1)
				rem[sl] = True
				rem[i] = False

		ind = np.arange(len(phi))[~rem]

		#resort
		ind_sorted = ind[np.argsort(phi[ind])][::-1]

	return ind_sorted

def _phi_hat(eps, mu, sigma, height):
	'''
	Calculates phi hat for given parameters.
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
	Calculates the difference between phi and phi_hat for scipy least_squares
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
	Class for storing f(Ea) and calculating peak deconvolution
	'''

	def __init__(self, eps, phi, nPeaks='auto', thres=0.05, combine_last=None):
		'''
		Initializes the EnergyComplex object.
		'''

		#assert phi and eps are same length
		if len(phi) != len(eps):
			raise ValueError('phi and eps vectors must have same length')
		nE = len(phi)

		#perform deconvolution
		mu,sigma,height = _deconvolve(eps, phi, nPeaks = nPeaks, thres=thres)
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
