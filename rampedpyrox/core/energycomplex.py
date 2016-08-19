import numpy as np



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

	def deconvolve(self, nPeaks = 'auto', combine_last = None):
		'''
		Deconvolves f(Ea) into Gaussian peaks.
		'''

	def plot():
		'''
		Plots the inverse or peak-deconvolved EC.
		'''

	def summary():
		'''
		Prints a summary of the EnergyComplex object.
		'''
