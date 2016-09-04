'''
This module contains the RateData superclass and all corresponding subclasses.

TODO: Store peak info
TODO: FIX IMPORT ERROR!!
'''

#import modules
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import norm

#import container classes
from rampedpyrox.core.array_classes import(
	rparray
	)

#import other rampedpyrox classes
# from rampedpyrox.timedata.timedata import(
# 	RpoThermogram,
# 	)
# from rampedpyrox.timedata.timedata import *

#import helper functions
from rampedpyrox.ratedata.ratedata_helper import(
	_deconvolve,
	)

from rampedpyrox.model.model_helper import(
	_calc_phi,
	)

class RateData(object):
	'''
	Class to store rate-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self):
		raise NotImplementedError

	#define classmethod to generate instance by inverse modeling
	@classmethod
	def inverse_model(cls, model, timedata, combined = None, nPeaks = 'auto',
		omega = 'auto', peak_shape = 'Gaussian', thres = 0.05, **kwargs):
		'''
		Inverse models a ``TimeData`` instance using a given ``Model``
		instance and creates a ``RateData`` instance.

		Parameters
		----------

		Keyword Arguments
		-----------------


		'''

		#extract model rate/Ea and store as k variable (necessary since models
		#	have different nomenclature)
		if hasattr(model, 'k'):
			k = model.k
		elif hasattr(model, 'Ea'):
			k = model.Ea

		#create class instance
		rd = cls(k, **kwargs)

		#calculate best-fit omega if necessary
		if omega == 'auto':
			omega = model.calc_L_curve(timedata, plot = False, **kwargs)

		#generate model-estimated pdf, phi
		phi, _, _ = _calc_phi(model, timedata, omega)

		#deconvolve into individual peaks
		peaks, peak_info = _deconvolve(k, phi, 
			nPeaks = nPeaks, 
			peak_shape = peak_shape,
			thres = thres)

		#combine peaks if necessary
		if combined is not None:
			peaks = peaks

		#input estimated data
		rd.input_estimated(peaks, peak_info, model.model_type, 
			omega = omega, 
			**kwargs)

		return rd

	#define a method to input estimated rate data
	def input_estimated(self, peaks, peak_info, model_type, omega = None,
		**kwargs):
		
		#pop acceptable kwargs:
		#	sig_figs

		sf = kwargs.pop('sig_figs', None)
		if kwargs:
			raise TypeError(
				'Unexpected **kwargs: %r' % kwargs)

		#extract n rate/Ea (necessary since models have different nomenclature)
		if hasattr(self, 'nk'):
			nk = self.nk
		elif hasattr(self, 'nEa'):
			nk = self.nEa

		#ensure type and size
		peaks = rparray(peaks, nk, sig_figs=sf)

		#force to be 2d (for derivatives and sums, below)
		nPeak = int(peaks.size/nk)
		peaks = peaks.reshape(nk, nPeak)

		#store attributes
		self.dof = nk - 3*nPeak
		self.model_type = model_type
		self.nPeak = nPeak
		self.peaks = peaks
		self.peak_info = peak_info

		#calculate phi and store
		self.phi = np.sum(peaks, axis=1)

		#input omega if it exists for bookkeeping
		if omega is not None:
			if not isinstance(omega, (int, float)):
				raise TypeError('omega must be None, int, or float')
			else:
				self.omega = omega

		#store statistics if the model has true data, f
		if hasattr(self, 'f'):

			rcs = norm((self.f - phi)/self.f_std)/self.dof
			rmse = norm(self.f - phi)/nk**0.5

			self.red_chi_sq = rcs
			self.rmse = rmse

	def plot(self, ax=None, labs=None, md=None, rd=None):
		raise NotImplementedError

	def summary(self):
		raise NotImplementedError


class EnergyComplex(RateData):
	__doc__='''
	Class for inputting, storing, and deconvolving Ramped PryOx activation
	energy distributions.

	Parameters
	----------

	Keyword Arguments
	-----------------

	Raises
	------

	Warnings
	--------

	Notes
	-----

	See Also
	--------

	Examples
	--------

	Attributes
	----------

	References
	----------
	'''

	def __init__(self, Ea, f = None, f_std = 0, **kwargs):

		#pop acceptable kwargs:
		#	sig_figs

		sf = kwargs.pop('sig_figs', None)
		if kwargs:
			raise TypeError(
				'Unexpected **kwargs: %r' % kwargs)

		#store activation energy attributes
		nEa = len(Ea)
		self.Ea = rparray(Ea, nEa, sig_figs = sf)
		self.nEa = nEa

		#check if fEa and store
		if f is not None:
			self.f = rparray(f, nEa, sig_figs = sf)
			self.f_std = rparray(f_std, nEa, sig_figs = sf)


	@classmethod
	def inverse_model(cls, model, timedata, combined = None, nPeaks = 'auto',
		omega = 'auto', peak_shape = 'Gaussian', thres = 0.05, **kwargs):
		'''
		Generates an energy complex by inverting a ``TimeData`` instance using
		a given ``Model`` instance.

		Parameters
		----------

		Keyword Arguments
		-----------------

		Raises
		------

		Warnings
		--------

		Notes
		-----

		See Also
		--------
		'''

		#check that timedata is the right type
		# if not isinstance(timedata, (RpoThermogram)):
		# 	warnings.warn((
		# 	"Attempting to generate EnergyComplex using a timedata instance" 
		# 	"of class: %s. Consider using RpoThermogram timedata instance"
		# 	"instead." % repr(timedata)))

		ec = super(EnergyComplex, cls).inverse_model(model, timedata,
			combined = combined,
			nPeaks = nPeaks,
			omega = omega,
			peak_shape = peak_shape,
			thres = thres,
			**kwargs)

		return ec

	def input_estimated(self, peaks, peak_info, model_type, omega = None,
		**kwargs):
		'''
		Inputs estimated rate data into the ``EnergyComplex`` instance and
		calculates statistics.

		Paramters
		---------

		Keyword Arguments
		-----------------

		Warnings
		--------

		Raises
		------

		Notes
		-----

		See Also
		--------
		'''

		#warn if using isothermal model
		if model_type not in ['Daem']:
			warnings.warn((
				"Attempting to use isothermal model for RPO run!"
				"Model type: %s. Consider using non-isothermal model"
				"such as 'Daem' instead." % model_type))

		super(EnergyComplex, self).input_estimated(peaks, peak_info,
			model_type,
			omega = omega,
			**kwargs)


	# def plot(self,):

	# def summary(self,):







