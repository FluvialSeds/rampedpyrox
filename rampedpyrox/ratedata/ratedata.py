'''
This module contains the RateData superclass and all corresponding subclasses.
'''

#import modules
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import norm

#import helper functions
from rampedpyrox.core.core_functions import(
	assert_len,
	)

from rampedpyrox.ratedata.ratedata_helper import(
	_deconvolve,
	)

from rampedpyrox.model.model_helper import(
	_calc_f,
	)

from rampedpyrox.core.plotting_helper import(
	_rem_dup_leg,
	)

from rampedpyrox.core.summary_helper import(
	_energycomplex_peak_info
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
			Array of k/Ea values considered in the model.

		Keyword Arguments
		-----------------
		f : array-like
			Array of a pdf of the distribution of k/Ea values.

		f_std : 
			Array of the uncertainty in f.
		'''
		raise NotImplementedError

	#define classmethod to generate instance by inverse modeling timedata with
	# a model
	@classmethod
	def inverse_model(cls, model, timedata, nPeaks = 'auto',
		omega = 'auto', peak_shape = 'Gaussian', thres = 0.05):
		'''
		Inverse models a ``TimeData`` instance using a given ``Model``
		instance and creates a ``RateData`` instance.

		Parameters
		----------
		model : rp.Model
			``Model`` instance containing the A matrix to use for inversion.

		timedata : rp.TimeData
			``TimeData`` instance containing the timeseries data to invert.

		Keyword Arguments
		-----------------
		nPeaks : int or 'auto'
			Tells the program how many peaks to retain after deconvolution.
			Defaults to 'auto'.

		omega : scalar or 'auto'
			Smoothing weighting factor for Tikhonov regularization. Defaults
			to 'auto'.

		peak_shape : str
			Peak shape to use for deconvolved peaks. Acceptable strings are:
				'Gaussian'
				'(add more later)'
			Defaults to 'Gaussian'.

		thres : float
			Threshold for peak detection cutoff. `thres` is the relative 
			height of the global maximum under which no peaks will be 
			detected. Defaults to 0.05 (i.e. 5% of the highest peak).

		Raises
		------
		TypeError
			If `nPeaks` is not int or 'auto'.

		TypeError
			If `omega` is not scalar or 'auto'.

		ValueError
			If `peak_shape` is not an acceptable string.

		TypeError
			If `thres` is not a float.

		Warnings
		--------
		Warns if ``scipy.optimize.least_squares`` cannot converge on a 
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
		forward_model
			``TimeData`` method for forward-modeling a ``RateData`` instance
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
			raise TypeError('omega must be int, float, or "auto"')

		#generate regularized "true" pdf, f
		f, resid_rmse, rgh_rmse = _calc_f(model, timedata, omega)

		#create class instance
		rd = cls(k, f = f)

		#deconvolve into individual peaks
		peaks, peak_info = _deconvolve(k, f, 
			nPeaks = nPeaks, 
			peak_shape = peak_shape,
			thres = thres)

		#input estimated data
		rd.input_estimated(model.model_type, peaks, peak_info, 
			omega = omega,
			resid_rmse = resid_rmse,
			rgh_rmse = rgh_rmse)

		return rd

	#define a method to input estimated rate data
	def input_estimated(self, model_type, peaks, omega = None,
		resid_rmse = None, rgh_rmse = None):
		'''
		Inputs estimated data into a ``RateData`` instance.

		Parameters
		----------
		model_type :  str
			String of the model type used to calculate ratedata.
		
		peaks : np.ndarray
			2d array of the pdf of individual peaks at each rate/Ea point.

		Keyword Arguments
		-----------------
		omega : scalar or 'auto'
			Smoothing weighting factor for Tikhonov regularization. Defaults
			to 'auto'.

		resid_rmse : float
			Residual RMSE from inverse model.

		rgh_rmse : float
			Roughness RMSE from inverse model.

		Raises
		------
		TypeError
			If omega is not scalar or None.
		'''

		#extract n rate/Ea (necessary since models have different nomenclature)
		if hasattr(self, 'nk'):
			nk = self.nk
		elif hasattr(self, 'nEa'):
			nk = self.nEa

		#ensure type and size
		peaks = assert_len(peaks, nk)

		#force to be 2d (for derivatives and sums, below)
		nPeak = int(peaks.size/nk)
		peaks = peaks.reshape(nk, nPeak)

		#store attributes
		self.dof = nk - 3*nPeak
		self.model_type = model_type
		self.nPeak = nPeak
		self.peaks = peaks
		self.resid_rmse = resid_rmse
		self.rgh_rmse = rgh_rmse

		#calculate phi and store
		self.phi = np.sum(peaks, axis = 1)

		#input omega if it exists for bookkeeping
		if omega is not None:
			if not isinstance(omega, (int, float)):
				raise TypeError('omega must be None, int, or float')
			else:
				self.omega = omega

		#store statistics if the model has true data, f
		if hasattr(self, 'f'):

			rcs = norm((self.f - self.phi)/self.f_std)/self.dof
			rmse = norm(self.f - self.phi)/nk**0.5

			self.red_chi_sq = rcs
			self.rmse = rmse

	#define plotting method
	def plot(self, ax=None, labs=None, md=None, rd=None):
		'''
		Method for plotting ``RateData`` instance data.

		Keyword Arguments
		-----------------
		axis : matplotlib.axis or None
			Axis handle to plot on.

		labs : tuple
			Tuple of axis labels, in the form (x_label, y_label).

		md : tuple or None
			Tuple of modeled data, in the form 
			(x_data, sum_y_data, cmpt_y_data). Defaults to None.

		rd : tuple
			Tuple of real data, in the form (x_data, y_data).

		Returns
		-------
		ax : matplotlib.axis
			Updated axis handle containing data.
		'''

		#create axis if necessary and label
		if ax is None:
			_, ax = plt.subplots(1,1)

		#label axes
		ax.set_xlabel(labs[0])
		ax.set_ylabel(labs[1])

		#add real data if it exists
		if rd is not None:
			#plot real data
			ax.plot(rd[0], rd[1],
				linewidth=2,
				color='k',
				label=r'Real Data ($\omega$ = %.2f)' %self.omega)

		#add model-estimated data if it exists
		if md is not None:

			#plot the model-estimated total
			ax.plot(md[0], md[1],
				linewidth=2,
				color='r',
				label=r'Deconvolved Data ($\phi$)')

			#plot individual components as shaded regions
			for cpt in md[2].T:

				ax.fill_between(md[0], 0, cpt,
					color='k',
					alpha=0.2,
					label='Components (n = %.0f)' %self.nPeak)

		#remove duplicate legend entries
		han_list, lab_list = _rem_dup_leg(ax)
		
		ax.legend(han_list,lab_list, 
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
		Array of activation energy, in kJ/mol. Length nEa.

	Keyword Arguments
	-----------------
	f : None or array-like
		Array of the "true" (i.e. before being deconvolved into peaks)
		pdf of the Ea distribution. Length nEa. Defaults to None.

	f_std : scalar or array-like
		Standard deviation of `f`, with length nEa. Defaults to zeros. 

	Raises
	------
	TypeError
		If `Ea` is not array-like.

	TypeError
		If `f` is not None or array-like.

	TypeError
		If `f_std` is not scalar or array-like.

	ValueError
		If `f` or `f_std` are not length nEa.

	See Also
	--------
	Daem
		``Model`` subclass used to generate the Laplace transform for RPO
		data and translate between time- and Ea-space.

	RpoThermogram
		``TimeData`` subclass containing the time and fraction remaining data
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
			y = (1/np.sqrt(2*np.pi*sig**2))*np.exp(-(x-mu)**2/(2*sig**2))
			return y

		f = Gaussian(Ea, 150, 10)

		#create the instance
		ec = rp.EnergyComplex(Ea, f = f)

	Manually inputting estimated peak data to the above instance::

		#import additional modules
		import pandas as pd

		#add 5 percent Gaussian noise to f
		phi = 0.05*np.max(f)*np.random.randn(len(f))*f

		#say the data was generated using a Daem model
		model_type = 'Daem'

		#create a peak_info pd.series
		peak_info = pd.Series([150, 10, 1.0],
			index = ['mu', 'sigma', 'rel. area'])

		ec.input_estimated(model_type, phi, peak_info, 
			omega = None, 
			resid_rmse = None,
			rgh_rmse = None)

	Or, insteand run the inversion to generate an energy complex using an 
	``rp.RpoThermogram`` instance, tg, and a ``Daem`` instance, daem::

		#keeping defaults, not combining any peaks
		ec = rp.EnergyComplex(daem, tg, 
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
		ec = rp.EnergyComplex(daem, tg, 
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

		ec.peak_info()

	Attributes
	----------
	dof : int
		Degrees of freedom of model fit, defined as ``nEa - 3*nPeak``.

	Ea : np.ndarray
		Array of activation energy, in kJ/mol. Length nEa.

	f : np.ndarray
		Array of the "true" (i.e. before being deconvolved into peaks)
		pdf of the Ea distribution. Length nEa.

	f_std : 
		Standard deviation of `f`, with length nEa.

	model_type : str
		String of the model type used for inversion, for bookkeeping.

	nEa : int
		Number of Ea points.

	nPeak : int
		Number of Gaussian peaks in estimated energy complex before being 
		combined (*i.e.* number of components).

	omega : float
		Smoothing weighting factor for Tikhonov regularization.

	peak_info : pd.DataFrame
		``pd.DataFrame`` instance containing the deconvolved peak
		summary info: mu, sigma, height, relative area.

	peaks : np.ndarray
		Array of the estimated peaks. Shape [nEa x nPeak].

	phi : np.ndarray
		Array of the estimated pdf of the Ea distribution. Length nEa.

	red_chi_sq : float
		The reduced chi square metric for the model fit.

	resid_rmse : float
		The RMSE between the measured thermogram data and the estimated 
		thermogram using the "true" pdf of Ea, f. Used for determining the
		best-fit omega value.

	rgh_rmse :
		The roughness "RMSE". Used for determining best-fit omega value.

	rmse : float
		The RMSE between "true" and estimated energy complex.

	References
	----------
	\B. Cramer (2004) Methane generation from coal during open system 
	pyrolysis investigated by isotope specific, Gaussian distributed reaction
	kinetics. *Organic Geochemistry*, **35**, 379-392.

	D.C. Forney and D.H. Rothman (2012) Common structure in the heterogeneity
	of plant-matter decay. *Journal of the Royal Society Interface*, 
	rsif.2012.0122.

	D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
	respiration rates from decay time series. *Biogeosciences*, **9**,
	3601-3612.
	'''

	def __init__(self, Ea, f = None, f_std = 0):

		#store activation energy attributes
		nEa = len(Ea)
		self.Ea = assert_len(Ea, nEa)
		self.nEa = nEa

		#check if fEa and store
		if f is not None:
			self.f = assert_len(f, nEa)
			self.f_std = assert_len(f_std, nEa)

	#define classmethod to generate instance by inverse modeling timedata with
	# a model
	@classmethod
	def inverse_model(cls, model, timedata, combined = None, nPeaks = 'auto',
		omega = 'auto', peak_shape = 'Gaussian', thres = 0.05):
		'''
		Generates an energy complex by inverting a ``TimeData`` instance using
		a given ``Model`` instance.

		Parameters
		----------
		model : rp.Model
			``Model`` instance containing the A matrix to use for inversion.

		timedata : rp.TimeData
			``TimeData`` instance containing the timeseries data to invert.

		Keyword Arguments
		-----------------
		combined :  list of tuples or None
			Tells the program which peaks to combine when deconvolving the
			ratedata. Must be a list of tuples -- e.g. [(0,1), (4,5)] will
			combine peaks 0 and 1, and 4 and 5. Defaults to None.

		nPeaks : int or 'auto'
			Tells the program how many peaks to retain after deconvolution.
			Defaults to 'auto'.

		omega : scalar or 'auto'
			Smoothing weighting factor for Tikhonov regularization. Defaults
			to 'auto'.

		peak_shape : str
			Peak shape to use for deconvolved peaks. Acceptable strings are:
				'Gaussian'
				'(add more later)'
			Defaults to 'Gaussian'.

		thres : float
			Threshold for peak detection cutoff. `thres` is the relative 
			height of the global maximum under which no peaks will be 
			detected. Defaults to 0.05 (i.e. 5% of the highest peak).

		Raises
		------
		TypeError
			If `combined` is not a list of tuples or None.

		TypeError
			If `nPeaks` is not int or 'auto'.

		TypeError
			If `omega` is not scalar or 'auto'.

		ValueError
			If `peak_shape` is not an acceptable string.

		TypeError
			If `thres` is not a float.

		Warnings
		--------
		Warns if ``scipy.optimize.least_squares`` cannot converge on a 
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
		forward_model
			``TimeData`` method for forward-modeling a ``RateData`` instance
			using a particular model.
		'''

		#import other rampedpyrox classes
		from .. import timedata as td

		#check that timedata is the right type
		if not isinstance(timedata, (td.RpoThermogram)):
			warnings.warn((
			"Attempting to generate EnergyComplex using a timedata instance" 
			"of class: %s. Consider using RpoThermogram timedata instance"
			"instead." % repr(timedata)))

		ec = super(EnergyComplex, cls).inverse_model(model, timedata,
			nPeaks = nPeaks,
			omega = omega,
			peak_shape = peak_shape,
			thres = thres)

		#assert combined type
		if isinstance(combined, list):
			if not all([isinstance(n, tuple) for n in combined]):
				raise TypeError('Elements of `combined` must be tuples')

			elif not all([isinstance(i, int) for tup in combined for i in tup]):
				raise TypeError('Elements of tuples in `combined` must be int')

		elif combined is not None:
			raise TypeError('combined must be a list of tuples or None')

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

			ec.peaks = np.delete(pks, del_pks, axis = 1)

		return ec

	#define a method to input estimated rate data
	def input_estimated(self, model_type, peaks, peak_info, omega = None, 
		resid_rmse = None, rgh_rmse = None):
		'''
		Inputs estimated rate data into the ``EnergyComplex`` instance and
		calculates statistics.

		Parameters
		----------
		model_type :  str
			String of the model type used to calculate ratedata.
		
		peaks : np.ndarray
			2d array of the pdf of individual peaks at each rate/Ea point.

		peak_info : pd.DataFrame
			``pd.DataFrame`` instance containing the deconvolved peak
			summary info: mu, sigma, height, relative area.

		Keyword Arguments
		-----------------
		omega : scalar or 'auto'
			Smoothing weighting factor for Tikhonov regularization. Defaults
			to 'auto'.

		resid_rmse : float
			Residual RMSE from inverse model.

		rgh_rmse : float
			Roughness RMSE from inverse model.

		Raises
		------
		TypeError
			If omega is not scalar or None.

		Warnings
		--------
		Warns if attempting to input data that was generated with a model
		other than `Daem`.

		Notes
		-----
		`peak_info` stores the peak information **before** being combined!
		'''

		#warn if using isothermal model
		if model_type not in ['Daem']:
			warnings.warn((
				"Attempting to use isothermal model for RPO run!"
				"Model type: %s. Consider using non-isothermal model"
				"such as 'Daem' instead." % model_type))

		super(EnergyComplex, self).input_estimated(model_type, peaks,
			omega = omega,
			resid_rmse = resid_rmse,
			rgh_rmse = rgh_rmse)

		#input EnergyComplex peak info
		self.peak_info = _energycomplex_peak_info(self, peak_info)

	#define plotting method
	def plot(self, ax = None):
		'''
		Plots the true and model-estimated Ea pdf (including individual 
		peaks) against Ea.

		Keyword Arguments
		-----------------
		ax : None or matplotlib.axis
			Axis to plot on. If `None`, automatically creates a
			``matplotlip.axis`` instance to return. Defaults to None.

		Returns
		-------
		ax : matplotlib.axis
			Updated axis instance with plotted data.

		Notes
		-----
		Number of peaks declared in the legend is **before** peaks have been
		combined!
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

		ax = super(EnergyComplex, self).plot(ax = ax, 
			md = md,
			labs = labs, 
			rd = rd)

		return ax




