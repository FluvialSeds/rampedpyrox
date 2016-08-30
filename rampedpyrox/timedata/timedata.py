'''
This module contains the TimeData superclass and all corresponding subclasses
(e.g. RpoThermogram).
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from timedata_helper import(
	_rpo_extract_tg,
	)


class TimeData(object):
	'''
	Class to store time data. Intended for subclassing, do not call directly.
	'''

	def __init__(self, dgdt, dgdt_std, g, g_std, add_noise, nt, t, T):

		#set public attributes
		self.add_noise = add_noise
		self.dgdt = dgdt #fraction/second
		self.dgdt_std = dgdt_std #fraction/second
		self.g = g #fraction
		self.g_std = g_std #fraction
		self.nt = nt
		self.t = t #seconds
		self.T = T #Kelvin

	def __setattr__(self, name, value):
		'''
		Use lazy attributes to add modeled data later.
		'''

		if name in ('gam','dgamdt', 'dgamdT', 'peaks') \
			and len(value) != self.nt:
			raise ValueError('%s must have length nt' % name)

		super(TimeData, self).__setattr__(name, value)


#	def plot(self):

#	def summary(self):


class RpoThermogram(TimeData):
	__doc__='''
	Class for inputting and storing Ramped PyrOx true and modeled thermograms,
	and for calculating goodness of fit statistics.

	Parameters
	----------
	all_data : str or pd.DataFrame
		File containing thermogram data, either as a path string or 
		``pd.DataFrame`` instance.

	add_noise : boolean
		Indicates whether the program should add normally distributed noise 
		when initializing the instance. Used with ``ppm_CO2_err`` and 
		``T_err`` for Monte Carlo simulations. Defaults to False.

	nt : int
		The number of time points to use. Defaults to 250.

	ppm_CO2_err : int or float
		The CO2 concentration standard deviation, in ppm. Used with 
		``add_noise`` for Monte Carlo simulations. Defaults to 5.

	T_err : int or float
		The temperature standard deviation, in Kelvin. Used with ``add_noise``
		for Monte Carlo simulations. Defaults to 3.

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
	add_noise : boolean
		Indicates whether the program should add normally distributed noise 
		when initializing the instance. Used with ``ppm_CO2_err`` and 
		``T_err`` for Monte Carlo simulations.

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

	dgdt_std : np.ndarray
		Standard deviation of `dgdt`. Length nt.	

	dgdT_std : np.ndarray
		Standard deviation of `dtdT`. Length nt.

	dof : int
		Degrees of freedom of model fit, defined as ``nt - 3*nPeak``.

	dTdt : np.ndarray
		Array of the derivative of temperature with respect to time (*i.e.*
		the ramp rate) at each timepoint, in Kelvin/second. Length nt.

	g : np.ndarray
		Array of the true fraction of carbon remaining at each timepoint.
		Length nt.

	gam : np.ndarray
		Array of the estimated fraction of carbon remaining at each timepoint.
		Length nt.

	g_std : np.ndarray
		Standard deviation of `g`. Length nt.

	model_type : str
		The inverse model used to calculate estimated thermogram.

	nt : int
		Number of timepoints.

	ppm_CO2_err : int or float
		The CO2 concentration standard deviation, in ppm. Used with 
		``add_noise`` for Monte Carlo simulations.

	red_chi_sq : float
		The reduced chi square metric for the model fit.

	RMSE : float
		The RMSE between true and estimated thermogram.

	t : np.ndarray
		Array of timep, in seconds. Length nt.

	T : np.ndarray
		Array of temperature, in Kelvin. Length nt.

	T_err : int or float
		The temperature standard deviation, in Kelvin. Used with ``add_noise``
		for Monte Carlo simulations. Defaults to 3.
	'''

	def __init__(self, all_data, add_noise=False, nt=250, ppm_CO2_err=5, 
		T_err=3):

		#extract arrays from `all_data`
		dgdt, dgdt_std, g, g_std, t, T = _rpo_extract_tg(all_data, nt)
		dTdt = np.gradient(T)/np.gradient(t)

		#initialize the superclass
		super(RpoThermogram, self).__init__(dgdt, dgdt_std, g, g_std,
			add_noise, nt, t, T)

		#set additional public attributes
		self.dgdT = dgdt/dTdt #fration/Kelvin
		self.dTdt = dTdt #Kelvin/second
		self.ppm_CO2_err = ppm_CO2_err #ppm
		self.T_err = T_err #Kelvin











