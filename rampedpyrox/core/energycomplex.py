import numpy as np

from numpy.linalg import norm

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

	return phi_hat

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


class EnergyComplex(object):
	'''
	Class for storing f(Ea) and calculating peak deconvolution
	'''

	def __init__(self, phi, eps):
		'''
		Initializes the EnergyComplex object.
		'''

		#assert phi and eps are same length
		if len(phi) != len(eps):
			raise ValueError('phi and eps vectors must have same length')

		#define public parameters
		self.phi = phi
		self.eps = eps

	def deconvolve(self, nPeaks = 'auto', thres=0.05, combine_last = None):
		'''
		Deconvolves f(Ea) into Gaussian peaks.
		'''

		#find peak indices
		if nPeaks is 'auto':
			ind_sorted = _peak_indices(self.phi, thres=thres, min_dist=1)
		elif isinstance(nPeaks,int):
			all_ind = _peak_indices(self.phi, thres=thres, min_dist=1)

			#check if nPeaks is greater than the total amount of peaks
			if len(all_ind) < nPeaks:
				raise ValueError('nPeaks > total detected at current threshold')
			
			ind_sorted = all_ind[:nPeaks]
		else:
			raise ValueError('nPeaks must be "auto" or int')

		#calculate arrays of each best-fit peak
		peaks = _fit_peaks(self.eps, self.phi, ind_sorted, )


	def plot():
		'''
		Plots the inverse or peak-deconvolved EC.
		'''

	def summary():
		'''
		Prints a summary of the EnergyComplex object.
		'''
