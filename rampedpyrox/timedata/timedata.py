'''
This module contains the TimeData superclass and all corresponding subclasses.
'''

#import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.linalg import norm

from rampedpyrox.timedata.timedata_helper import(
	_assert_lent,
	_derivatize,
	_rpo_extract_tg,
	)


class TimeData(object):
	'''
	Class to store time-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, t, T, **kwargs):

		#store attributes
		nt = len(t)
		self.nt = nt
		self.t = np.array(t) #s
		self.T = _assert_lent(T, nt) #K

		#unpack keyword only args
		g = kwargs.pop('g', np.zeros(nt)) #frac
		g_std = kwargs.pop('g_std', np.zeros(nt)) #frac
		T_std = kwargs.pop('T_std', np.zeros(nt)) #K
		if kwargs:
			raise TypeError('Unexpected **kwargs: %r' %kwargs)

		#store keyword only args as attributes
		self.g = _assert_lent(g, nt) #fraction
		self.g_std = _assert_lent(g_std, nt) #fraction
		self.T_std = _assert_lent(T_std, nt) #K

		#store calculated attributes
		self.dgdt = _derivatize(g,t)
		self.dgdT = _derivatize(g,T)
		self.dTdt = _derivatize(T,t)

	#define class method for creating instance directly from .csv file
	@classmethod
	def from_csv(cls, file):
		raise NotImplementedError

	#define method for inputting the results from a model fit
	def input_estimated(self, cmpt, model_type):
		'''
		Method to input modeled estimate data into ``TimeData`` instance and
		calculate corresponding statistics.
		'''
		
		#check inputted data types
		if not isinstance(cmpt, (list, np.ndarray)):
			raise TypeError('cmpt must be array-like.')

		elif not isinstance(model_type, str):
			raise TypeError('model_type must be string')
		
		elif isinstance(cmpt, list):
			#ensure ndarray
			cmpt = np.array(cmpt)

		#check dimensionality and make 2d if necessary
		if cmpt.ndim == 1:
			cmpt = np.reshape(cmpt, (len(cmpt), 1))

		#check length is nt
		nt, nPeak = np.shape(cmpt)
		if nt != self.nt:
			raise ValueError('cmpt array must have length nt')

		#store attributes
		self.dof = nt - 3*nPeak
		self.model_type = model_type
		self.nPeak = nPeak
		self.cmpt = cmpt

		#generate necessary arrays
		gam = np.sum(cmpt, axis=1)
		pt = [_derivatize(col, self.t) for col in cmpt.T]
		pT = [_derivatize(col, self.T) for col in cmpt.T]

		#calculate derived attributes and store
		self.dgamdt = _derivatize(gam, self.t)
		self.dgamdT = _derivatize(gam, self.T)
		self.dcmptdt = np.column_stack(pt)
		self.dcmptdT = np.column_stack(pT)
		self.gam = gam
		self.red_chi_sq = norm((self.g - gam)/self.g_std)/self.dof
		self.RMSE = norm(self.g - gam)/nt**0.5
	
	def plot(self):
		raise NotImplementedError

	def summary(self):
		raise NotImplementedError


class RpoThermogram(TimeData):
	__doc__='''
	Class for inputting and storing Ramped PyrOx true and modeled thermograms,
	and for calculating goodness of fit statistics.

	Parameters
	----------
	t : array-like
		Array of timep, in seconds. Length nt.

	T : array-like
		Array of temperature, in Kelvin. Length nt.

	Keyword Arguments
	-----------------
	g : array-like or None
		Array of the true fraction of carbon remaining at each timepoint,
		with length nt. Defaults to zeros.

	g_std : array-like or None
		Standard deviation of `g`, with length nt. Defaults to zeros.

	T_std : array-like or None
		The temperature standard deviation, with length nt, in Kelvin. Used
		for Monte Carlo simulations. Defaults to zeros.

	Raises
	------

	Notes
	-----

	See Also
	--------

	Examples
	--------

	Attributes
	----------
	cmpt : np.ndarray
		Array of the estimated fraction of carbon remaining in each component 
		at each timepoint. Shape [nt x nPeak].

	dcmptdt : np.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		in each component with respect to time at each timepoint, in 
		fraction/second. Shape [nt x nPeak].

	dcmptdT : np.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		in each component with respect to temperature at each timepoint, in 
		fraction/Kelvin. Shape [nt x nPeak].

	dgamdt : np.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		with respect to time at each timepoint, in fraction/second. Length nt.

	dgamdT : np.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		with respect to temperature at each timepoint, in fraction/Kelvin.
		Length nt.

	dgdt : np.ndarray
		Array of the derivative of the true fraction of carbon remaining with
		respect to time at each timepoint, in fraction/second. Length nt.

	dgdT : np.ndarray
		Array of the derivative of the true fraction of carbon remaining with 
		respect to temperature at each timepoint, in fraction/Kelvin.
		Length nt.

	dof : int
		Degrees of freedom of model fit, defined as ``nt - 3*nPeak``.

	dTdt : np.ndarray
		Array of the derivative of temperature with respect to time (*i.e.*
		the instantaneous ramp rate) at each timepoint, in Kelvin/second.
		Length nt.

	g : np.ndarray
		Array of the true fraction of carbon remaining at each timepoint.
		Length nt.

	g_std : np.ndarray
		Array of the standard deviation of `g`. Length nt.

	gam : np.ndarray
		Array of the estimated fraction of carbon remaining at each timepoint.
		Length nt.

	model_type : str
		The inverse model used to calculate estimated thermogram.

	nPeak : int
		Number of Gaussian peaks in estimated thermogram (*i.e.* number of 
		components)

	nt : int
		Number of timepoints.

	red_chi_sq : float
		The reduced chi square metric for the model fit.

	RMSE : float
		The RMSE between true and estimated thermogram.

	t : np.ndarray
		Array of timep, in seconds. Length nt.

	T : np.ndarray
		Array of temperature, in Kelvin. Length nt.

	T_std : np.ndarray
		Array of the standard deviation of `T`. Length nt.
	'''

	def __init__(self, t, T, **kwargs):

		#ensure T is array-like
		if not isinstance(T, (list, np.ndarray)):
			raise TypeError('RPO run must contain temp array (NOT scalar!)')

		super(RpoThermogram, self).__init__(t, T, **kwargs)

	#define class method for creating instance directly from .csv file
	@classmethod
	def from_csv(cls, file, nt=250, ppm_CO2_err=5, T_err=3):
		'''
		Class method to directly import RPO data from a .csv file and create
		an ``RpoThermogram`` class instance.

		Parameters
		----------
		file : str or pd.DataFrame
			File containing thermogram data, either as a path string or 
			``pd.DataFrame`` instance.

		Keyword Arguments
		-----------------
		nt : int
			The number of time points to use. Defaults to 250.

		ppm_CO2_err : int or float
			The CO2 concentration standard deviation, in ppm. Used to 
			calculate `g_std`. Defaults to 5.

		T_err : int or float
			The uncertainty in the RPO instrument temperature, in Kelvin.
			Used to calculate `T_std` array. Defaults to 3.
		
		Raises
		------
		TypeError
			If `file` is not str or ``pd.DataFrame`` instance.
		
		TypeError
			If index of `file` is not ``pd.DatetimeIndex`` instance.

		TypeError
			If `nt` is not int.

		ValueError
			If `file` does not contain "CO2_scaled" and "temp" columns.

		Notes
		-----
		If using the `all_data` file generated by the NOSAMS RPO LabView 
		program, the date_time column must be converted to **hh:mm:ss AM/PM**
		format and a header row must be added with the following columns:

		date_time, T_room, P_room, CO2_raw, corr_int, corr_slope, temp, 
		CO2_scaled flow_rate, dTdt, fraction, ug_frac, ug_sum

		Note that all columns besides `date_time`, `temp`, and `CO2_scaled`
		are unused. Ensure that all rows before the start of temperature
		ramping and after the ovens have been turned off have been removed.

		When down-sampling, `t` contains the midpoints of each time bin and
		`g` and `T` contain the corresponding temp. and g at each midpoint.

		See Also
		--------

		Examples
		--------

		'''

		#extract data from file
		g, g_std, t, T = _rpo_extract_tg(file, nt, ppm_CO2_err)

		return cls(t, T, g=g, g_std=g_std, T_std=T_err)

	#define method for inputting model-estimate data
	def input_estimated(self, cmpt, model_type):
		'''
		Inputs the results of an inverse model into the ``RpoThermogram``
		instance and calculates statistics.
		
		Parameters
		----------
		cmpt : array-like

		model_type : str

		Warnings
		--------
		Raises warning if using an isothermal model type for an RPO run.

		Raises
		------
		TypeError
			If `cmpt` is not array-like.

		TypeError
			If `model_type` is not a string.

		ValueError
			If `cmpt` is not of length nt.

		Notes
		-----

		See Also
		--------

		Examples
		--------
		'''

		#warn if using isothermal model
		if model_type not in ('Daem'):
			warnings.warn('Attempting to use isothermal model for RPO run!')

		super(RpoThermogram, self).input_estimated(cmpt, model_type)

