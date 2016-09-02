'''
This module contains the TimeData superclass and all corresponding subclasses.

* TODO: Add summary method
* TODO: Plotting kwargs
'''

import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.linalg import norm

#import container classes
from rampedpyrox.core.array_classes import(
	rparray
	)

#import package-level functions
from rampedpyrox.core.core_functions import(
	derivatize
	)

#import helper functions
from rampedpyrox.timedata.timedata_helper import(
	_plot_dicts,
	_rem_dup_leg,
	_rpo_extract_tg,
	)


class TimeData(object):
	'''
	Class to store time-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, t, T, g=0, g_std=0, T_std=0, **kwargs):
		'''
		Initialize the superclass.

		Parameters
		----------
		t : array-like
			Array of timep, in seconds. Length nt.

		T : array-like
			Array of temperature, in Kelvin. Length nt.

		Keyword Arguments
		-----------------
		g : scalar or array-like
			Array of the true fraction of carbon remaining at each timepoint,
			with length nt. Defaults to zeros.

		g_std : scalar or array-like
			Standard deviation of `g`, with length nt. Defaults to zeros.

		T_std : scalar or array-like
			The temperature standard deviation, with length nt, in Kelvin. 
			Used for Monte Carlo simulations. Defaults to zeros.

		sig_figs : int
			The number of significant figures to use for g, g_std, etc. 
			arrays. Defaults to 10.
		'''

		#pop acceptable kwargs:
		#	sig_figs

		sf = kwargs.pop('sig_figs', 10)
		if kwargs:
			raise TypeError(
				'Unexpected **kwargs: %r' % kwargs)

		#store attributes
		nt = len(t)
		self.nt = nt
		self.t = rparray(t, nt, sig_figs=1) #s
		self.T = rparray(T, nt, sig_figs=3) #K
		self.g = rparray(g, nt, sig_figs=sf) #fraction
		self.g_std = rparray(g_std, nt, sig_figs=sf) #fraction
		self.T_std = rparray(T_std, nt, sig_figs=3) #K

		#store calculated attributes
		self.dgdt = self.g.derivatize(t, sig_figs=sf)
		self.dgdT = self.g.derivatize(T, sig_figs=sf)
		self.dTdt = self.T.derivatize(t, sig_figs=3)

	#define class method for creating instance directly from .csv file
	@classmethod
	def from_csv(cls, file):
		raise NotImplementedError

	#define method for creating ratedata instance
	# i.e. "running the inverse model"
	def calc_ratedata(cls, model):
		raise NotImplementedError

	#define method for inputting the results from a model fit
	def input_estimated(self, cmpt, model_type, **kwargs):
		'''
		Method to input modeled estimate data into ``TimeData`` instance and
		calculate corresponding statistics.

		Parameters
		----------
		cmpt : array-like
			Array of fraction of each component remaining at each timestep.
			Gets converted to 2d rparray.

		model_type : str
			String of the model type used. Acceptable values:
				Daem

		Keyword Arguments
		-----------------
		sig_figs : int
			The number of significant figures to use for g, g_std, etc. 
			arrays. Defaults to 10.
		'''

		#pop acceptable kwargs:
		#	sig_figs

		sf = kwargs.pop('sig_figs', 10)
		if kwargs:
			raise TypeError(
				'Unexpected **kwargs: %r' % kwargs)

		#check model_type type
		if not isinstance(model_type, str):
			raise TypeError(
				'model_type must be string')

		#ensure type and size
		nt = self.nt
		cmpt = rparray(cmpt, nt, sig_figs=sf)

		#force to be 2d (for derivatives and sums, below)
		nPeak = int(cmpt.size/nt)
		cmpt = cmpt.reshape(nt, nPeak)

		#store attributes
		self.dof = nt - 3*nPeak
		self.model_type = model_type
		self.nPeak = nPeak
		self.cmpt = cmpt

		#generate necessary arrays
		gam = np.sum(cmpt, axis=1)
		rcs = norm((self.g - gam)/self.g_std)/self.dof
		rmse = norm(self.g - gam)/nt**0.5

		#calculate derived attributes and store
		self.dgamdt = gam.derivatize(self.t, sig_figs=sf)
		self.dgamdT = gam.derivatize(self.T, sig_figs=sf)
		self.dcmptdt = cmpt.derivatize(self.t, sig_figs=sf)
		self.dcmptdT = cmpt.derivatize(self.T, sig_figs=sf)
		self.gam = gam
		self.red_chi_sq = rcs
		self.rmse = rmse
	
	#define plotting method
	def plot(self, ax=None, labs=None, md=None, rd=None, **kwargs):
		'''
		Method for plotting ``TimeData`` instance data.

		Keyword Arguments
		-----------------
		axis : mpl.axishandle or None
			Axis handle to plot on.

		labs : tuple
			Tuple of axis labels, in the form (x_label, y_label).

		md : tuple or None
			Tuple of modeled data, in the form 
			(x_data, sum_y_data, cmpt_y_data). Defaults to None.

		rd : tuple
			Tuple of real data, in the form (x_data, y_data).
		'''

		#create axis if necessary
		if ax is None:
			_, ax = plt.subplots(1,1)

		#plot real data
		ax.plot(rd[0], rd[1],
			linewidth=2,
			color='k',
			label='Real Data')

		#label axes
		ax.set_xlabel(labs[0])
		ax.set_ylabel(labs[1])

		#add model-estimated data if it exists
		if md is not None:

			#plot the model-estimated total
			ax.plot(md[0], md[1],
				linewidth=1.5,
				color='r',
				label='Modeled Data')

			#plot individual components as shaded regions
			for cpt in md[2].T:

				ax.fill_between(md[0], 0, cpt,
					color='k',
					alpha=0.2,
					label='Components (n=%.0f)' %self.nPeak,
					**kwargs)

		#remove duplicate legend entries
		han_list, lab_list = _rem_dup_leg(ax)
		
		ax.legend(han_list,lab_list, 
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
		Array of time, in seconds. Length nt.

	T : array-like
		Array of temperature, in Kelvin. Length nt.

	Keyword Arguments
	-----------------
	g : scalar or array-like
		Array of the true fraction of carbon remaining at each timepoint,
		with length nt. Defaults to zeros.

	g_std : scalar or array-like
		Standard deviation of `g`, with length nt. Defaults to zeros.

	T_std : scalar or array-like
		The temperature standard deviation, with length nt, in Kelvin. Used
		for Monte Carlo simulations. Defaults to zeros.

	sig_figs : int
		The number of significant figures to use for g, g_std, etc. arrays.
		Defaults to 10.

	Raises
	------
	TypeError
		If `t` is not array-like.

	TypeError
		If `g`, `g_std`, `T`, or `T_std` are not scalar or array-like.

	ValueError
		If any of `T`, `g`, `g_std`, or `T_std` are not length nt.

	Warnings
	--------
	If attempting to use isothermal data to create an ``RpoThermogram``
	instance.

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

		#plot resulting rates against time and temp
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
	cmpt : rp.rparray
		Array of the estimated fraction of carbon remaining in each component 
		at each timepoint. Shape [nt x nPeak].

	dcmptdt : rp.rparray
		Array of the derivative of the estimated fraction of carbon remaining
		in each component with respect to time at each timepoint, in 
		fraction/second. Shape [nt x nPeak].

	dcmptdT : rp.rparray
		Array of the derivative of the estimated fraction of carbon remaining
		in each component with respect to temperature at each timepoint, in 
		fraction/Kelvin. Shape [nt x nPeak].

	dgamdt : rp.rparray
		Array of the derivative of the estimated fraction of carbon remaining
		with respect to time at each timepoint, in fraction/second. Length nt.

	dgamdT : rp.rparray
		Array of the derivative of the estimated fraction of carbon remaining
		with respect to temperature at each timepoint, in fraction/Kelvin.
		Length nt.

	dgdt : rp.rparray
		Array of the derivative of the true fraction of carbon remaining with
		respect to time at each timepoint, in fraction/second. Length nt.

	dgdT : rp.rparray
		Array of the derivative of the true fraction of carbon remaining with 
		respect to temperature at each timepoint, in fraction/Kelvin.
		Length nt.

	dof : int
		Degrees of freedom of model fit, defined as ``nt - 3*nPeak``.

	dTdt : rp.rparray
		Array of the derivative of temperature with respect to time (*i.e.*
		the instantaneous ramp rate) at each timepoint, in Kelvin/second.
		Length nt.

	g : rp.rparray
		Array of the true fraction of carbon remaining at each timepoint.
		Length nt.

	g_std : rp.rparray
		Array of the standard deviation of `g`. Length nt.

	gam : rp.rparray
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

	t : rp.rparray
		Array of timep, in seconds. Length nt.

	T : rp.rparray
		Array of temperature, in Kelvin. Length nt.

	T_std : rp.rparray
		Array of the standard deviation of `T`. Length nt.
	'''

	def __init__(self, t, T, g=0, g_std=0, T_std=0, **kwargs):

		#warn if T is scalar
		if isinstance(T, (int, float)):
			warnings.warn((
				"Attempting to use isothermal data for RPO run! T is a scalar"
				"value of: %.1f. Consider using an isothermal model type" 
				"instead." % T))

		super(RpoThermogram, self).__init__(t, T, 
			g=g, 
			g_std=g_std, 
			T_std=T_std, 
			*kwargs)

	#define class method for creating instance directly from .csv file
	@classmethod
	def from_csv(cls, file, nt=250, ppm_CO2_err=5, T_err=3, **kwargs):
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

		sig_figs : int
			The number of significant figures to use for g, g_std, etc. 
			arrays. Defaults to 10.
		
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

		return cls(t, T, g=g, g_std=g_std, T_std=T_err, **kwargs)

	#define method for creating ratedata instance
	# i.e. "running the inverse model"
	def calc_ratedata(cls, model):
		raise NotImplementedError

	#define method for inputting model-estimate data
	def input_estimated(self, cmpt, model_type, **kwargs):
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

		Keyword Arguments
		-----------------
		sig_figs : int
			The number of significant figures to use for g, g_std, etc. 
			arrays. Defaults to 10.

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
			warnings.warn((
				"Attempting to use isothermal model for RPO run!"
				"Model type: %s. Consider using non-isothermal model"
				"such as 'Daem' instead." % model_type))

		super(RpoThermogram, self).input_estimated(cmpt, model_type, 
			**kwargs)

	#define plotting method
	def plot(self, ax=None, xaxis='time', yaxis='rate', **kwargs):
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

		**kwargs:
			Matplotlib **kwargs.

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

		#check that axes are appropriate strings
		if xaxis not in ['time','temp']:
			raise ValueError((
				"xaxis does not accept %r."
				"Must be either 'time' or 'temp'" %xaxis))

		elif yaxis not in ['fraction','rate']:
			raise ValueError((
				"yaxis does not accept %r."
				"Must be either 'rate' or 'fraction'" %yaxis))

		#convert `xaxis` and `yaxis` to approrpiate dicts, extract data
		rpo_rd = _plot_dicts('rpo_rd', self)
		rpo_labs = _plot_dicts('rpo_labs', self)

		rd = (rpo_rd[xaxis][yaxis][0], rpo_rd[xaxis][yaxis][1])
		labs = (rpo_labs[xaxis][yaxis][0], rpo_labs[xaxis][yaxis][1])

		#check if modeled data exist
		if hasattr(self, 'cmpt'):
			#extract modeled data dict
			rpo_md = _plot_dicts('rpo_md', self)
			md = (rd[0], rpo_md[xaxis][yaxis][0], rpo_md[xaxis][yaxis][1])

		else:
			md = None

		ax = super(RpoThermogram, self).plot(ax=ax, 
			md=md,
			labs=labs, 
			rd=rd,
			**kwargs)

		return ax










