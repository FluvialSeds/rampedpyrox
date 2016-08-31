'''
This module contains the TimeData superclass and all corresponding subclasses.

* TODO: Add summary method
'''

import matplotlib.pyplot as plt
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
		rcs = norm((self.g - gam)/self.g_std)/self.dof
		rmse = norm(self.g - gam)/nt**0.5

		#calculate derived attributes and store
		self.dgamdt = _derivatize(gam, self.t)
		self.dgamdT = _derivatize(gam, self.T)
		self.dcmptdt = np.column_stack(pt)
		self.dcmptdT = np.column_stack(pT)
		self.gam = gam
		self.red_chi_sq = rcs
		self.RMSE = rmse
	
	#define plotting method
	def plot(self, ax=None, xaxis='time', yaxis='rate'):
		'''
		Method for plotting ``TimeData`` instance data.
		'''
		
		#check that axis are appropriate strings
		if xaxis not in ['time','temp']:
			raise ValueError('xaxis does not accept %r.' \
				'Must be either "time" or "temp"' %xaxis)

		elif yaxis not in ['fraction','rate']:
			raise ValueError('yaxis does not accept %r.' \
				'Must be either "rate" or "fraction"' %yaxis)

		#create axis if necessary
		if ax is None:
			_, ax = plt.subplots(1,1)

		#create a nested dict to keep track of cases for real data
		rd = {'time': {'fraction' : (self.t, self.g),
						'rate' : (self.t, -self.dgdt)},
			'temp': {'fraction' : (self.T, self.g),
						'rate' : (self.T, -self.dgdT)}}

		#create a nested dict to keep track of axis labels
		labs = {'time': {'fraction' : ('time (s)', 'g (unitless)'),
						'rate' : ('time (s)', r'fraction/time $(s^{-1})$')},
			'temp' : {'fraction' : ('temp (K)', 'g (unitless)'),
						'rate' : ('temp (K)', r'fraction/temp $(K^{-1})$')}}

		#plot real data
		ax.plot(rd[xaxis][yaxis][0], rd[xaxis][yaxis][1],
			linewidth=2,
			color='k',
			label='Real Data')

		#label axes
		ax.set_xlabel(labs[xaxis][yaxis][0])
		ax.set_ylabel(labs[xaxis][yaxis][1])

		#add model-estimated data if it exists
		if hasattr(self, 'cmpt'):

			#create a nested dict to keep track of cases of modeled data
			md = {'time': {'fraction' : (self.gam, self.cmpt),
							'rate' : (-self.dgamdt, -self.dcmptdt)},
				'temp': {'fraction' : (self.gam, self.cmpt),
							'rate' : (-self.dgamdT, -self.dcmptdT)}}

			#plot the model-estimated total
			ax.plot(rd[xaxis][yaxis][0], md[xaxis][yaxis][0],
				linewidth=1.5,
				color='r',
				label='Modeled Data')

			#plot individual components as shaded regions
			for cpt in md[xaxis][yaxis][1].T:

				ax.fill_between(rd[xaxis][yaxis][0], 0, cpt,
					color='k',
					alpha=0.2,
					label='Components (n=%.0f)' %self.nPeak)

		#remove duplicate legend entries
		han, lab = ax.get_legend_handles_labels()
		h_list, l_list = [], []
		
		for h, l in zip(han, lab):
			if l not in l_list:
				h_list.append(h)
				l_list.append(l)
		
		ax.legend(h_list,l_list, 
			loc='best',
			frameon=False)

		return ax

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
	TypeError
		If `t` and `T` are not array-like.

	TypeError
		If `g`, `g_std`, or `T_std` are not scalar or array-like.

	ValueError
		If any of `T`, `g`, `g_std`, or `T_std` are not length nt.

	Notes
	-----

	See Also
	--------
	Daem
		``Model`` subclass used to generate the Laplace transform for RPO
		data and translate between time- and Ea-space.

	EnergyComplex
		``RateData`` subclass for storing, deconvolving, and analyzing RPO
		rate data.

	Examples
	--------
	Generating a bare-bones thermogram containing only `t` and `T`::

		#import modules
		import numpy as np
		import rampedpyrox as rp

		#generate arbitrary data
		t = np.arange(1,100) #100 second experiment
		beta = 0.5 #K/second
		T = beta*t + 273.15 #K

		#create instance
		tg = rp.RpoThermogram(t,T)

	This bare-bones thermogram can be used later to project a ``Daem``
	instance onto any arbitrary time-temperature history.

	Generating a real thermogram using an RPO output .csv file and the
	``RpoThermogram.from_csv`` class method::

		#import modules
		import rampedpyrox as rp

		#create path to data file
		file = 'path_to_folder_containing_data/data.csv'

		#create instance
		tg = rp.RpoThermogram.from_csv(file,
			nt = 250,
			ppm_CO2_err = 5,
			T_err = 3)

	Manually adding some model-estimated component data as `cmpt`::

		#assuming cmpt has been generating using a Daem model
		tg.input_estimated(cmpt, 'Daem')

	Plotting the resulting true and estimated thermograms::

		#import additional modules
		import matplotlib.pyplot as plt

		#create figure
		fig, ax = plt.subplots(1,2)

		#plot resultint rates against time and temp
		ax[0] = tg.plot(ax = ax[0], 
			xaxis = 'time', 
			yaxis = 'rate')
		
		ax[1] = tg.plot(ax = ax[1], 
			xaxis = 'temp', 
			yaxis = 'rate')

	Generating a summary of the analysis::

		tg.summary()

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
			Array of the estimated fraction of carbon remaining in each 
			component at each timepoint. Shape [nt x nPeak].

		model_type : str
			The type of inverse model used to generate estimate data. Warns
			if an isothermal model.

		Warnings
		--------
		Warns if using an an isothermal model type for an RPO run.

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
		'''

		#warn if using isothermal model
		if model_type not in ['Daem']:
			warnings.warn('Attempting to use isothermal model for RPO run!')

		super(RpoThermogram, self).input_estimated(cmpt, model_type)

	#define plotting method
	def plot(self, ax=None, xaxis='time', yaxis='rate'):
		'''
		Plots the true and model-estimated thermograms (including individual 
		peaks) against time or temp.

		Keyword Arguments
		-----------------
		ax : None or matplotlib.axis
			Axis to plot on. If `None`, automatically creates a
			``matplotlip.axis`` instance to return. Defaults to None.

		xaxis : str
			Sets the x axis unit, either 'time' or 'temp'. Defaults to 'time'.

		yaxis : str
			Sets the y axis unit, either 'fraction' or 'rate'. Defaults to 
			'rate'.

		Returns
		-------
		ax : matplotlib.axis
			Updated axis instance with plotted data.

		Raises
		------
		ValueError
			If `xaxis` is not 'time' or 'temp'.

		ValueError
			if `yaxis` is not 'fraction' or 'rate'.
		'''

		ax = super(RpoThermogram, self).plot(ax=ax, xaxis=xaxis, yaxis=yaxis)

		return ax
















