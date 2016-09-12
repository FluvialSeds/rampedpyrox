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
from ..core.exceptions import(
	ArrayError,
	ScalarError,
	)

#import helper functions
from ..core.core_functions import(
	assert_len,
	)

from ..core.plotting_helper import(
	_rem_dup_leg,
	)

from ..core.summary_helper import(
	_energycomplex_peak_info,
	)

from .ratedata_helper import(
	_deconvolve,
	)

from ..model.model_helper import(
	_calc_f,
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
			Array of k/Ea values considered in the model. Length `nk`.

		f : array-like
			Array of a pdf of the distribution of k/Ea values. Length `nk`.

		f_std : 
			Array of the uncertainty in f. Length `nk`.
		'''
		raise NotImplementedError

	#define classmethod to generate instance by inverse modeling timedata with
	# a model
	@classmethod
	def inverse_model(
			cls, 
			model, 
			timedata, 
			nPeaks = 'auto',
			omega = 'auto', 
			peak_shape = 'Gaussian', 
			thres = 0.05):
		'''
		Inverse models an ``rp.TimeData`` instance using a given ``rp.Model``
		instance and creates an ``rp.RateData`` instance.

		Parameters
		----------
		model : rp.Model
			``rp.Model`` instance containing the A matrix to use for inversion.

		timedata : rp.TimeData
			``rp.TimeData`` instance containing the timeseries data to invert.

		nPeaks : int or 'auto'
			Tells the program how many peaks to retain after deconvolution.
			Defaults to 'auto'.

		omega : scalar or 'auto'
			Tikhonov regularization weighting factor. Defaults to 'auto'.

		peak_shape : str
			Peak shape to use for deconvolved peaks. Acceptable strings are:

				'Gaussian'
			
			Defaults to 'Gaussian'.

		thres : float
			Threshold for peak detection cutoff. `thres` is the relative 
			height of the global maximum under which no peaks will be 
			detected. Defaults to 0.05 (i.e. 5% of the highest peak).

		Raises
		------
		ScalarError
			If `omega` is not scalar or 'auto'.

		Warnings
		--------
		UserWarning
			If ``scipy.optimize.least_squares`` cannot converge on a 
			solution.

		Notes
		-----
		This method calculates peaks according to changes in curvature in the
		`f` array resulting from the inverse model. Each bounded section 
		with a negative second derivative (i.e. concave down) and `f` value 
		above `thres` is considered a unique peak. If `nPeaks` is not 'auto', 
		these peaks are sorted according to decreasing peak heights and the 
		first `nPeaks` peaks are saved.

		See Also
		--------
		TimeData.forward_model
			``rp.TimeData`` method for forward-modeling an ``rp.RateData`` instance
			using a particular model.
		'''

		#extract model rate/Ea and store as k variable (necessary since models
		#	have different nomenclature)
		if hasattr(model, 'k'):
			k = model.k
		
		elif hasattr(model, 'Ea'):
			k = model.Ea

		#calculate best-fit omega if necessary
		if omega == 'auto':
			omega = model.calc_L_curve(timedata, plot = False)
		
		elif isinstance(omega, (int, float)):
			omega = float(omega)
		
		else:
			raise ScalarError(
				'omega must be int, float, or "auto"')

		#generate regularized "true" pdf, f
		f, resid_rmse, rgh_rmse = _calc_f(model, timedata, omega)

		#create class instance
		rd = cls(k, f = f)

		#deconvolve into individual peaks
		peaks, peak_info = _deconvolve(
			k, 
			f, 
			nPeaks = nPeaks, 
			peak_shape = peak_shape,
			thres = thres)

		#input estimated data
		rd.input_estimated(
			peaks, 
			omega = omega,
			peak_info = peak_info,
			peak_shape = peak_shape,
			resid_rmse = resid_rmse,
			rgh_rmse = rgh_rmse)

		return rd

	#define a method to input estimated rate data
	def input_estimated(
			self,
			peaks, 
			omega = None, 
			peak_info = None, 
			peak_shape = 'Gaussian', 
			resid_rmse = None, 
			rgh_rmse = None):
		'''
		Inputs estimated data into an ``rp.RateData`` instance.

		Parameters
		----------		
		peaks : np.ndarray
			2d array of the pdf of individual peaks at each rate/Ea point.

		peak_info : np.ndarray
			2d array of peak mean, stdev., and height

		omega : scalar or 'auto'
			Smoothing weighting factor for Tikhonov regularization. Defaults
			to 'auto'.

		peak_shape : str
			Peak shape to use for deconvolved peaks. Acceptable strings are:

				'Gaussian'
			
			Defaults to 'Gaussian'.

		resid_rmse : float
			Residual RMSE from inverse model.

		rgh_rmse : float
			Roughness RMSE from inverse model.

		Raises
		------
		ScalarError
			If omega is not scalar or None.

		Warnings
		--------
		UserWarning
			If peaks do not integrate to one, the program automatically scales
			all peaks equally to ensure proper integration value.
		'''

		#extract n rate/Ea (necessary since models have different nomenclature)
		if hasattr(self, 'nk'):
			nk = self.nk
			k = self.k
		
		elif hasattr(self, 'nEa'):
			nk = self.nEa
			k = self.Ea

		#ensure type and size
		peaks = assert_len(peaks, nk)

		#force to be 2d (for derivatives and sums, below)
		nPeak = int(peaks.size/nk)
		peaks = peaks.reshape(nk, nPeak)

		#ensure that peaks integrate to one, and warn if not
		phi = np.sum(peaks, axis = 1)
		a = np.sum(phi*np.gradient(k))

		if np.around(a - 1, decimals = 2) != 0:
			warnings.warn(
				'Peaks do not integrate to one with 1 percent precision.'
				' Integral is: %r. Automatically scalaing to unity.' %a,
				UserWarning)

			peaks = peaks/a

		#store attributes
		self.dof = nk - 3*nPeak + 1
		self.nPeak = nPeak
		self.peak_shape = peak_shape
		self.peaks = peaks
		self.phi = phi
		self.resid_rmse = resid_rmse
		self.rgh_rmse = rgh_rmse

		#store protected _pkinf attribute (used for isotope calcs.)
		self._pkinf = peak_info

		#input omega if it exists for bookkeeping
		if omega is not None:
			if not isinstance(omega, (int, float)):
				raise ScalarError(
					'omega must be None, int, or float')
			
			else:
				self.omega = omega

		#store statistics if the model has true data, f
		if hasattr(self, 'f'):

			rcs = norm((self.f - self.phi)/self.f_std)/self.dof
			rmse = norm(self.f - self.phi)/nk**0.5

			self.red_chi_sq = rcs
			self.rmse = rmse

	#define plotting method
	def plot(self, ax = None, labs = None, md = None, rd = None):
		'''
		Method for plotting ``rp.RateData`` instance data.

		Parameters
		----------
		axis : matplotlib.axis or None
			Axis handle to plot on. Defaults to `None`.

		labs : tuple
			Tuple of axis labels, in the form (x_label, y_label).
			Defaults to `None`.

		md : tuple or None
			Tuple of modeled data, in the form 
			(x_data, sum_y_data, cmpt_y_data). Defaults to `None`.

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
				linewidth=2,
				color='k',
				label=r'Real Data ($\omega$ = %.2f)' %self.omega)

			#set limits
			ax.set_ylim([0, 1.1*np.max(rd[1])])

		#add model-estimated data if it exists
		if md is not None:

			#plot the model-estimated total
			ax.plot(
				md[0], 
				md[1],
				linewidth=2,
				color='r',
				label=r'Deconvolved Data ($\phi$)')

			#plot individual components as shaded regions
			for cpt in md[2].T:

				ax.fill_between(
					md[0], 0, 
					cpt,
					color='k',
					alpha=0.2,
					label='Components (n = %.0f)' %self.nPeak)

		#remove duplicate legend entries
		han_list, lab_list = _rem_dup_leg(ax)
		
		ax.legend(
			han_list,
			lab_list, 
			loc='best',
			frameon=False)

		return ax


class EnergyComplex(RateData):
	__doc__='''
	Class for inputting, storing, and deconvolving Ramped PryOx activation
	energy distributions.

	Parameters
	----------
	Ea : array-like
		Array of activation energy, in kJ/mol. Length `nEa`.

	f : None or array-like
		Array of the "true" (i.e. before being deconvolved into peaks)
		pdf of the Ea distribution. Length `nEa`. Defaults to `None`.

	f_std : scalar or array-like
		Standard deviation of `f`, with length `nEa`. Defaults to zeros. 

	Raises
	------
	ArrayError
		If the any value in `Ea` is negative.

	See Also
	--------
	Daem
		``rp.Model`` subclass used to generate the Laplace transform for RPO
		data and translate between time- and Ea-space.

	RpoThermogram
		``rp.TimeData`` subclass containing the time and fraction remaining data
		used for the inversion.

	Examples
	--------
	Generating a bare-bones energy complex containing only `Ea` and `f`::

		#import modules
		import rampedpyrox as rp
		import numpy as np

		#generate arbitrary Gaussian data
		Ea = np.arange(50, 350)

		def Gaussian(x, mu, sig):
			scalar = (1/np.sqrt(2*np.pi*sig**2))*
			y = scalar*np.exp(-(x-mu)**2/(2*sig**2))
			return y

		f = Gaussian(Ea, 150, 10)

		#create the instance
		ec = rp.EnergyComplex(Ea, f = f)

	Manually inputting estimated peak data to the above instance::

		#import additional modules
		import pandas as pd

		#add 5 percent Gaussian noise to f
		phi = 0.05*np.max(f)*np.random.randn(len(f))*f

		#create a peak_info pd.series
		peak_info = pd.Series([150, 10, 1.0],
			index = ['mu', 'sigma', 'rel. area'])

		ec.input_estimated(
			phi, 
			peak_info, 
			omega = None, 
			resid_rmse = None,
			rgh_rmse = None)

	Or, insteand run the inversion to generate an energy complex using an 
	``rp.RpoThermogram`` instance, tg, and an ``rp.Daem`` instance, daem::

		#keeping defaults, not combining any peaks
		ec = rp.EnergyComplex(
			daem, 
			tg, 
			combined = None, 
			nPeaks = 'auto',
			omega = 'auto', 
			peak_shape = 'Gaussian', 
			thres = 0.05)

	Same as above, but now setting `omega` and combining peaks::

		#set values
		omega = 3
		combined = [(0,1), (6,7)]

		#create the instance
		ec = rp.EnergyComplex(
			daem, 
			tg, 
			combined = combined, 
			nPeaks = 'auto',
			omega = omega, 
			peak_shape = 'Gaussian', 
			thres = 0.05)

	Plotting the resulting "true" and estimated energy complex::

		#import additional modules
		import matplotlib.pyplot as plt

		#create figure
		fig, ax = plt.subplots(1,1)

		#plot resulting Ea pdfs.
		ax = ec.plot(ax = ax)

	Printing a summary of the analysis::

		print(ec.peak_info)

	**Attributes**

	dof : int
		Degrees of freedom of model fit, defined as ``nEa - 3*nPeak + 1``.

	Ea : np.ndarray
		Array of activation energy, in kJ/mol. Length `nEa`.

	f : np.ndarray
		Array of the "true" (i.e. before being deconvolved into peaks)
		pdf of the Ea distribution. Length `nEa`.

	f_std : 
		Standard deviation of `f`, with length `nEa`.

	nEa : int
		Number of Ea points.

	nPeak : int
		Number of Gaussian peaks in estimated energy complex.

	omega : float
		Tikhonov regularization weighting factor.

	peak_info : pd.DataFrame
		Dataframe containing the inverse-modeled peak isotope summary info: 

			mu (kJ/mol), \n
			sigma (kJ/mol), \n
			height (unitless), \n
			relative area

	peaks : np.ndarray
		Array of the estimated peaks. Shape [`nEa` x `nPeak`].

	phi : np.ndarray
		Array of the estimated pdf of the Ea distribution. Length `nEa`.

	red_chi_sq : float
		The reduced chi square metric for the model fit.

	resid_rmse : float
		The RMSE between the measured thermogram data and the estimated 
		thermogram using the "true" pdf of Ea, f. Used for determining the
		best-fit omega value.

	rgh_rmse :
		The roughness RMSE. Used for determining best-fit omega value.

	rmse : float
		The RMSE between "true" and estimated energy complex.

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

	def __init__(self, Ea, f = None, f_std = 0):

		#store activation energy attributes
		nEa = len(Ea)

		#ensure types
		Ea = assert_len(Ea, nEa)

		#ensure that Ea is non-negative
		if np.min(Ea) < 0:
			raise ArrayError(
				'Minimum value for Ea is: %r. Elements in Ea must be'
				' non-negative.' % np.min(Ea))

		self.Ea = Ea
		self.nEa = nEa
		
		#create protected _cmbd attribute to store combined peaks
		self._cmbd = None

		#check if fEa and store
		if f is not None:
			self.f = assert_len(f, nEa)
			self.f_std = assert_len(f_std, nEa)

	#define classmethod to generate instance by inverse modeling timedata with
	# a model
	@classmethod
	def inverse_model(
			cls, 
			model, 
			timedata, 
			combined = None, 
			nPeaks = 'auto',
			omega = 'auto', 
			peak_shape = 'Gaussian', 
			thres = 0.05):
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

		combined :  list of tuples or None
			Tells the program which peaks to combine when deconvolving the
			ratedata. Must be a list of tuples -- e.g. [(0,1), (4,5)] will
			combine peaks 0 and 1, and 4 and 5. Defaults to `None`.

		nPeaks : int or 'auto'
			Tells the program how many peaks to retain after deconvolution.
			Defaults to 'auto'.

		omega : scalar or 'auto'
			Smoothing weighting factor for Tikhonov regularization. Defaults
			to 'auto'.

		peak_shape : str
			Peak shape to use for deconvolved peaks. Acceptable strings are:

				'Gaussian'
			
			Defaults to 'Gaussian'.

		thres : float
			Threshold for peak detection cutoff. `thres` is the relative 
			height of the global maximum under which no peaks will be 
			detected. Defaults to 0.05 (i.e. 5% of the highest peak).

		Raises
		------
		ArrayError
			If `combined` is not a list of tuples or `None`.

		ArrayError
			If the elements of `combined` are not tuples.

		ScalarError
			If the elements of the tuples in `combined` are not int.

		Warnings
		--------
		UserWarning
			If ``scipy.optimize.least_squares`` cannot converge on a solution.

		UserWarning
			If attempting to use timedata that is not a ``rp.RpoThermogram``
			instance.

		UserWarning
			If attempting to use a model that is not a ``rp.Daem`` instance.
		
		Notes
		-----
		This method calculates peaks according to changes in curvature in the
		`f` array resulting from the inverse model. Each bounded section 
		with a negative second derivative (i.e. concave down) and `f` value 
		above `thres` is considered a unique peak. If `nPeaks` is not 'auto', 
		these peaks are sorted according to decreasing peak heights and the 
		first `nPeaks` peaks are saved.

		See Also
		--------
		RpoThermogram.forward_model
			``rp.TimeData`` method for forward-modeling an ``rp.RateData`` 
			instance using a particular model.

		References
		----------
		[1] B. de Caprariis et al. (2012) Double-Gaussian distributed activation
  			energy model for coal devolatilization. *Energy & Fuels*, **26**,
 			 6153-6159.
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
			nPeaks = nPeaks,
			omega = omega,
			peak_shape = peak_shape,
			thres = thres)

		#assert combined type
		if isinstance(combined, list):
			if not all([isinstance(n, tuple) for n in combined]):
				raise ArrayError('Elements of `combined` must be tuples')

			elif not all([isinstance(i, int) for tup in combined for i in tup]):
				raise ScalarError('Elements of tuples in `combined` must be int')

		elif combined is not None:
			raise ArrayError('combined must be a list of tuples or None')

		#combine peaks if necessary
		if combined is not None:
			
			#sum rows and put in list
			pks = ec.peaks
			del_pks = []
			
			for tup in combined:
				#subtract 1 to get into python indexing
				c = list([x - 1 for x in tup])

				#sum over combined columns and replace first col with sum
				pks[:, c[0]] = np.sum(pks[:,c], axis = 1)

				#store column indices to delete
				del_pks.append(c[1:])

			#flatten del_pks
			del_pks = [item for sl in del_pks for item in sl]

			ec.peaks = np.delete(pks, del_pks, axis = 1)

			#store combined as protected attribute
			ec._cmbd = del_pks

		return ec

	#define a method to input estimated rate data
	def input_estimated(
			self, 
			peaks, 
			omega = 0, 
			peak_info = None, 
			peak_shape = 'Gaussian', 
			resid_rmse = 0, 
			rgh_rmse = 0):
		'''
		Inputs estimated rate data into the ``rp.EnergyComplex`` instance and
		calculates statistics.

		Parameters
		----------
		peaks : np.ndarray
			2d array of the pdf of individual peaks at each rate/Ea point.

		peak_info : None or pd.DataFrame
			Dataframe containing the inverse-modeled peak isotope summary info: 

				mu (kJ/mol), \n
				sigma (kJ/mol), \n
				height (unitless), \n
				relative area

			Defaults to `None`.

		omega : scalar
			Tikhonov regularization weighting factor used to generate
			estimated data. Defaults to 0.

		peak_shape : str
			Peak shape to use for deconvolved peaks. Acceptable strings are:
				
				'Gaussian'

			Defaults to 'Gaussian'.

		resid_rmse : float
			Residual RMSE for the inputted estimated data. Defaults to 0.

		rgh_rmse : float
			Roughness RMSE for the inputted estimated data. Defaults to 0.
		'''

		super(EnergyComplex, self).input_estimated(
			peaks,
			omega = omega,
			peak_info = peak_info,
			peak_shape = peak_shape,
			resid_rmse = resid_rmse,
			rgh_rmse = rgh_rmse)

		#input EnergyComplex peak info
		if peak_info is not None:
			self.peak_info = _energycomplex_peak_info(self)

	#define plotting method
	def plot(self, ax = None):
		'''
		Plots the true and model-estimated Ea pdf (including individual 
		peaks) against Ea.

		Keyword Arguments
		-----------------
		ax : None or matplotlib.axis
			Axis to plot on. If `None`, automatically creates a
			``matplotlip.axis`` instance to return. Defaults to `None`.

		Returns
		-------
		ax : matplotlib.axis
			Updated axis instance with plotted data.

		Notes
		-----
		Number of peaks declared in the legend is **before** being combined!
		'''

		#create axis label tuple
		labs = (r'Ea (kJ/mol)', r'f(Ea) pdf (unitless)')

		#check if real data exist
		if hasattr(self, 'f'):
			#extract real data
			rd = (self.Ea, self.f)
		else:
			rd = None

		#check if modeled data exist
		if hasattr(self, 'peaks'):
			#extract modeled data dict
			md = (self.Ea, self.phi, self.peaks)
		else:
			md = None

		ax = super(EnergyComplex, self).plot(
			ax = ax, 
			md = md,
			labs = labs, 
			rd = rd)

		return ax

if __name__ == '__main__':

	import rampedpyrox as rp
