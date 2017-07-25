'''
This module contains the RateData superclass and all corresponding subclasses.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['EnergyComplex']

#import modules
import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.linalg import norm

#import exceptions
from .exceptions import(
	ArrayError,
	ScalarError,
	)

#import helper functions
from .core_functions import(
	assert_len,
	)

from .plotting_helper import(
	_rem_dup_leg,
	)

from .summary_helper import(
	_calc_rate_info,
	)

from .model_helper import(
	_calc_p,
	)

class RateData(object):
	'''
	Class to store rate-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self):
		'''
		Initialize the superclass

		Parameters
		----------
		k : array-like
			Array of k/E values considered in the model. Length `nk`.

		p : array-like
			Array of a pdf of the distribution of k/E values. Length `nk`.
		'''
		raise NotImplementedError

	#define classmethod to generate instance by inverse modeling timedata with
	# a given model
	@classmethod
	def inverse_model(
			cls, 
			model, 
			timedata, 
			lam = 'auto'):
		'''
		Inverse models an ``rp.TimeData`` instance using a given ``rp.Model``
		instance and creates an ``rp.RateData`` instance.

		Parameters
		----------
		model : rp.Model
			``rp.Model`` instance containing the A matrix to use for inversion.

		timedata : rp.TimeData
			``rp.TimeData`` instance containing the timeseries data to invert.

		Raises
		------
		ScalarError
			If `lam` is not scalar or 'auto'.

		Warnings
		--------
		UserWarning
			If ``scipy.optimize.least_squares`` cannot converge on a 
			solution.

		See Also
		--------
		TimeData.forward_model
			``rp.TimeData`` method for forward-modeling an ``rp.RateData`` instance
			using a particular model.
		'''

		#extract model rate/E and store as k variable (necessary since models
		# have different nomenclature)
		if hasattr(model, 'k'):
			k = model.k
		
		elif hasattr(model, 'E'):
			k = model.E

		#calculate best-fit lambda value if necessary
		if lam in ['auto', 'Auto']:
			lam = model.calc_L_curve(timedata, plot = False)
		
		elif isinstance(lam, (int, float)):
			lam = float(lam)
		
		else:
			raise ScalarError(
				'lam must be int, float, or "auto"')

		#generate regularized pdf, p
		p, resid, rgh = _calc_p(model, timedata, lam)

		#create class instance
		rd = cls(k, p = p)

		#input estimated data
		rd.input_estimated(
			lam = lam,
			resid = resid,
			rgh = rgh)

		return rd

	#define a method to input estimated rate data
	def input_estimated(
			self,
			lam = None, 
			resid = None, 
			rgh = None):
		'''
		Inputs estimated data into an ``rp.RateData`` instance.

		Parameters
		----------		
		lam : scalar
			Best-fit smoothing weighting factor for Tikhonov regularization.
			Calculated using L-curve approach.

		resid : float
			Residual RMSE from inverse model.

		rgh : float
			Roughness from inverse model.

		Raises
		------
		ScalarError
			If lam is not scalar or `None`.
		'''

		#extract n rate/E (necessary since models have different nomenclature)
		if hasattr(self, 'nk'):
			nk = self.nk
			k = self.k
		
		elif hasattr(self, 'nE'):
			nk = self.nE
			k = self.E

		#store attributes
		self.resid = resid
		self.rgh = rgh

		#input lam if it exists for bookkeeping
		if lam is not None:
			if not isinstance(lam, (int, float)):
				raise ScalarError(
					'lam must be None, int, or float')
			
			else:
				self.lam = lam

	#define plotting method
	def plot(self, ax = None, labs = None, rd = None):
		'''
		Method for plotting ``rp.RateData`` instance data.

		Parameters
		----------
		axis : matplotlib.axis or None
			Axis handle to plot on. Defaults to `None`.

		labs : tuple
			Tuple of axis labels, in the form (x_label, y_label).
			Defaults to `None`.

		rd : tuple
			Tuple of real data, in the form (x_data, y_data).

		Returns
		-------
		ax : matplotlib.axis
			Updated axis handle containing data.
		'''

		#create axis if necessary and label
		if ax is None:
			_, ax = plt.subplots(1, 1)

		#label axes
		if labs is not None:
			ax.set_xlabel(labs[0])
			ax.set_ylabel(labs[1])

		#add real data if it exists
		if rd is not None:
			#plot real data
			ax.plot(
				rd[0], 
				rd[1],
				linewidth = 2,
				color = 'k',
				label = r'Regularized p(0,E) ($\lambda$ = %.2f)' %self.lam)

			#set limits
			ax.set_xlim([0, 1.1*np.max(rd[0])])
			ax.set_ylim([0, 1.1*np.max(rd[1])])

		#remove duplicate legend entries
		han_list, lab_list = _rem_dup_leg(ax)
		
		ax.legend(
			han_list,
			lab_list, 
			loc = 'best',
			frameon = False)

		#make tight layout
		plt.tight_layout()

		return ax


class EnergyComplex(RateData):
	__doc__='''
	Class for inputting and storing Ramped PryOx activation energy 
	distributions.

	Parameters
	----------
	E : array-like
		Array of activation energy, in kJ/mol. Length `nE`.

	p : None or array-like
		Array of the regularized pdf of the E distribution, p(0,E). Length
		`nE`. Defaults to `None`.

	Raises
	------
	ArrayError
		If the any value in `E` is negative.

	See Also
	--------
	Daem
		``rp.Model`` subclass used to generate the Laplace transform for RPO
		data and translate between time- and E-space.

	RpoThermogram
		``rp.TimeData`` subclass containing the time and fraction remaining data
		used for the inversion.

	Examples
	--------
	Generating a bare-bones energy complex containing only `E` and `p`::

		#import modules
		import rampedpyrox as rp
		import numpy as np

		#generate arbitrary Gaussian data
		E = np.arange(50, 350)

		def Gaussian(x, mu, sig):
			scalar = (1/np.sqrt(2*np.pi*sig**2))*
			y = scalar*np.exp(-(x-mu)**2/(2*sig**2))
			return y

		p = Gaussian(E, 150, 10)

		#create the instance
		ec = rp.EnergyComplex(E, p = p)

	Or, insteand run the inversion to generate an energy complex using an 
	``rp.RpoThermogram`` instance, tg, and an ``rp.Daem`` instance, daem::

		#keeping defaults, not combining any peaks
		ec = rp.EnergyComplex(
			daem, 
			tg, 
			lam = 'auto')

	Plotting the resulting regularized energy complex::

		#import additional modules
		import matplotlib.pyplot as plt

		#create figure
		fig, ax = plt.subplots(1,1)

		#plot resulting E pdf, p(0,E)
		ax = ec.plot(ax = ax)

	**Attributes**

	E : np.ndarray
		Array of activation energy, in kJ/mol. Length `nE`.

	nE : int
		Number of E points.

	ec_info : pd.Series
		Series containing the observed EnergyComplex summary info: 

			E_max (kJ/mol), \n
			E_mean (kJ/mol), \n
			E_std (kJ/mol), \n
			p0E_max

	lam : float
		Tikhonov regularization weighting factor.

	p : np.ndarray
		Array of the pdf of the E distribution, p0E. Length `nE`.

	resid : float
		The RMSE between the measured thermogram data and the estimated 
		thermogram using the p (ghat). Used for determining the best-fit lambda
		value.

	rgh :
		The roughness RMSE. Used for determining best-fit lambda value.

	References
	----------
	[1] B. Cramer (2004) Methane generation from coal during open system 
		pyrolysis investigated by isotope specific, Gaussian distributed 
		reaction kinetics. *Organic Geochemistry*, **35**, 379-392.

	[2] D.C. Forney and D.H. Rothman (2012) Common structure in the 
		heterogeneity of plant-matter decay. *Journal of the Royal Society*
		*Interface*, rsif.2012.0122.

	[3] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.
	'''

	def __init__(self, E, p = None):

		#store activation energy attributes
		nE = len(E)

		#ensure types
		E = assert_len(E, nE)

		#ensure that E is non-negative
		if np.min(E) < 0:
			raise ArrayError(
				'Minimum value for E is: %r. Elements in E must be'
				' non-negative.' % np.min(E))

		self.E = E
		self.nE = nE

		#check if p exists and store p, statistics
		if p is not None:
			self.p = assert_len(p, nE)
			self.ec_info = _calc_rate_info(E, p, kstr = 'E')

	#define classmethod to generate instance by inverse modeling timedata with
	# a model
	@classmethod
	def inverse_model(
			cls, 
			model, 
			timedata, 
			lam = 'auto'):
		'''
		Generates an energy complex by inverting an ``rp.TimeData`` instance 
		using a given ``rp.Model`` instance.

		Parameters
		----------
		model : rp.Model
			``rp.Model`` instance containing the A matrix to use for 
			inversion.

		timedata : rp.TimeData
			``rp.TimeData`` instance containing the timeseries data to invert.

		lam : scalar or 'auto'
			Smoothing weighting factor for Tikhonov regularization. Defaults
			to 'auto'.

		Warnings
		--------
		UserWarning
			If ``scipy.optimize.least_squares`` cannot converge on a solution.

		UserWarning
			If attempting to use timedata that is not a ``rp.RpoThermogram``
			instance.

		UserWarning
			If attempting to use a model that is not a ``rp.Daem`` instance.

		See Also
		--------
		RpoThermogram.forward_model
			``rp.TimeData`` method for forward-modeling an ``rp.RateData`` 
			instance using a particular model.
		'''

		#warn if model is not Daem
		mod_type = type(model).__name__

		if mod_type not in ['Daem']:
			warnings.warn(
				'Attempting to calculate isotopes using a model instance of'
				' type %r. Consider using rp.Daem instance instead'
				% rd_type, UserWarning)

		#warn if timedata is not RpoThermogram
		td_type = type(timedata).__name__

		if td_type not in ['RpoThermogram']:
			warnings.warn(
				'Attempting to calculate isotopes using an isothermal timedata'
				' instance of type %r. Consider using rp.RpoThermogram' 
				' instance instead' % td_type, UserWarning)

		ec = super(EnergyComplex, cls).inverse_model(
			model, 
			timedata,
			lam = lam)

		return ec

	#define a method to input estimated rate data
	def input_estimated(
			self, 
			lam = 0, 
			resid = 0, 
			rgh = 0):
		'''
		Inputs estimated rate data into the ``rp.EnergyComplex`` instance and
		calculates statistics.

		Parameters
		----------
		lam : scalar
			Tikhonov regularization weighting factor used to generate
			estimated data. Defaults to 0.

		resid : float
			Residual RMSE for the inputted estimated data. Defaults to 0.

		rgh : float
			Roughness RMSE for the inputted estimated data. Defaults to 0.
		'''

		super(EnergyComplex, self).input_estimated(
			lam = lam,
			resid = resid,
			rgh = rgh)

	#define plotting method
	def plot(self, ax = None):
		'''
		Plots the pdf of E, p(0,E), against E.

		Keyword Arguments
		-----------------
		ax : None or matplotlib.axis
			Axis to plot on. If `None`, automatically creates a
			``matplotlip.axis`` instance to return. Defaults to `None`.

		Returns
		-------
		ax : matplotlib.axis
			Updated axis instance with plotted data.
		'''

		#create axis label tuple
		labs = (r'E (kJ/mol)', r'$p(0, E)$')

		#check if data exist
		if hasattr(self, 'p'):
			#extract data
			rd = (self.E, self.p)
		else:
			rd = None

		ax = super(EnergyComplex, self).plot(
			ax = ax, 
			labs = labs, 
			rd = rd)

		return ax

if __name__ == '__main__':

	import rampedpyrox as rp
