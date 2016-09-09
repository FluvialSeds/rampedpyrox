'''
This module contains the TimeData superclass and all corresponding subclasses.
'''

from __future__ import print_function

__docformat__ = 'restructuredtext en'
__all__ = ['RpoThermogram']

import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.linalg import norm

#import helper functions
from rampedpyrox.core.core_functions import(
	assert_len,
	derivatize,
	)

from rampedpyrox.timedata.timedata_helper import(
	_rpo_extract_tg,
	)

from rampedpyrox.model.model_helper import(
	_calc_cmpt
	)

from rampedpyrox.core.plotting_helper import(
	_plot_dicts,
	_rem_dup_leg,
	)

from rampedpyrox.core.summary_helper import(
	_timedata_peak_info
	)


class TimeData(object):
	'''
	Class to store time-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, t, T, g = None, g_std = 0, T_std = 0):
		'''
		Initialize the superclass.

		Parameters
		----------
		t : array-like
			Array of timepoints, in seconds. Length `nt`.

		T : array-like
			Array of temperature, in Kelvin. Length `nt`.

		g : None or array-like
			Array of the true fraction of carbon remaining at each timepoint,
			with length `nt`. Defaults to `None`.

		g_std : None or array-like
			Standard deviation of `g`, with length `nt`. Defaults to zero.

		T_std : scalar or array-like
			The temperature standard deviation, with length `nt`, in Kelvin. 
			Defaults to zero.

		Raises
		------
		TypeError
			If `t` is not array-like.

		TypeError
			If `g_std`, `T`, or `T_std` are not scalar or array-like.

		TypeError
			If `g` is not None or array-like.

		ValueError
			If any of `T`, `g`, `g_std`, or `T_std` are not length `nt`.
		'''

		#store time-temperature attributes
		nt = len(t)
		self.nt = nt
		self.t = assert_len(t, nt) #s
		self.T = assert_len(T, nt) #K
		self.T_std = assert_len(T_std, nt) #K

		#store time-temperature derivatives
		self.dTdt = derivatize(self.T, t) #K/s

		#check if g and store
		if g is not None:
			self.g = assert_len(g, nt) #fraction
			self.g_std = assert_len(g_std, nt) #fraction

			#store g derivatives
			self.dgdt = derivatize(g, t)
			self.dgdT = derivatize(g, T)

	#define class method for creating instance directly from .csv file
	@classmethod
	def from_csv(cls, file):
		raise NotImplementedError

	#define method for forward-modeling rate data using a given model
	def forward_model(self, model, ratedata):
		'''
		Forward-models rate data for a given model and creates timedata
		instance.

		Parameters
		----------
		model : rp.Model
			``rp.Model`` instance used to calculate the forward model.

		ratedata : rp.RateData
			``rp.Ratedata instance containing the reactive continuum data.

		Warnings
		--------
		UserWarning
			If the time-temperature data in the ``rp.Model`` instance do not 
			match the time-temperature data in the ``rp.TimeData`` instance.
		'''

		#warn if self and model t and T arrays do not match
		td_type = type(self).__name__
		mod_type = type(model).__name__

		if (self.t != model.t).any() or (self.T != model.T).any():
			warnings.warn(
				'rp.TimeTata instance of type %s and rp.Model instance of'
				' type %s do not contain matching time-temperature arrays.'
				' Check that the model does not correspond to a different'
				' rp.TimeData instance' %(td_type, mod_type))

		#extract components
		cmpt = _calc_cmpt(model, ratedata)

		#populate with modeled data
		self.input_estimated(cmpt)

	#define method for inputting the results from a model fit
	def input_estimated(self, cmpt):
		'''
		Method to input modeled estimate data into ``rp.TimeData`` instance 
		and to calculate corresponding statistics.

		Parameters
		----------
		cmpt : array-like
			Array of fraction of each component remaining at each timestep.
			Gets converted to 2d rparray. Shape [`nt` x `nPeak`].

		Raises
		------
		AttributeError
			If ``rp.TimeData`` instance does not contain necessary attributes
			(i.e. if it does not have inputted model-estimated data).
		'''

		#ensure type and size
		nt = self.nt
		cmpt = assert_len(cmpt, nt)

		#force to be 2d (for derivatives and sums, below)
		nPeak = int(cmpt.size/nt)
		cmpt = cmpt.reshape(nt, nPeak)

		#store attributes
		self.dof = nt - 3*nPeak + 1
		self.nPeak = nPeak
		self.cmpt = cmpt

		#generate gamma array
		gam = np.sum(cmpt, axis=1)

		#calculate derived attributes and store
		self.dgamdt = derivatize(gam, self.t)
		self.dgamdT = derivatize(gam, self.T)
		self.dcmptdt = derivatize(cmpt, self.t)
		self.dcmptdT = derivatize(cmpt, self.T)
		self.gam = gam

		#store statistics if the model has true data, g
		if hasattr(self, 'g'):

			rcs = norm((self.g - gam)/self.g_std)/self.dof
			rmse = norm(self.g - gam)/nt**0.5

			self.red_chi_sq = rcs
			self.rmse = rmse

		#store peak info
		self.peak_info = _timedata_peak_info(self)

	#define plotting method
	def plot(self, ax = None, labs = None, md = None, rd = None):
		'''
		Method for plotting ``rp.TimeData`` instance data.

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
			Tuple of real data, in the form (x_data, y_data). Defaults to
			`None`.

		Returns
		-------
		ax : matplotlib.axis
			Updated axis handle containing data.
		'''

		#create axis if necessary and label
		if ax is None:
			_, ax = plt.subplots(1, 1)

		#label axes if labels exist
		if labs is not None:
			ax.set_xlabel(labs[0])
			ax.set_ylabel(labs[1])

		#add real data if it exists
		if rd is not None:
			ax.plot(
				rd[0], 
				rd[1],
				linewidth=2,
				color='k',
				label='Real Data')

		#add model-estimated data if it exists
		if md is not None:

			#plot the model-estimated total
			ax.plot(
				md[0], 
				md[1],
				linewidth=2,
				color='r',
				label='Modeled data')

			#plot individual components as shaded regions
			for cpt in md[2].T:

				ax.fill_between(
					md[0], 
					0, 
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


class RpoThermogram(TimeData):
	__doc__='''
	Class for inputting and storing Ramped PyrOx true and modeled thermograms,
	and for calculating goodness of fit statistics.

	Parameters
	----------
	t : array-like
		Array of time, in seconds. Length `nt`.

	T : array-like
		Array of temperature, in Kelvin. Length `nt`.

	g : None or array-like
		Array of the true fraction of carbon remaining at each timepoint,
		with length `nt`. Defaults to `None`.

	g_std : scalar or array-like
		Standard deviation of `g`, with length `nt`. Defaults to zeros.

	T_std : scalar or array-like
		The temperature standard deviation, with length `nt`, in Kelvin. 
		Defaults to zeros.

	Raises
	------
	TypeError
		If `t` is not array-like.

	TypeError
		If `g_std`, `T`, or `T_std` are not scalar or array-like.

	TypeError
		If `g` is not None or array-like.

	ValueError
		If any of `T`, `g`, `g_std`, or `T_std` are not length `nt`.

	Warnings
	--------
	UserWarning
		If attempting to use isothermal data to create an ``rp.RpoThermogram``
		instance. Consider using alternate ``rp.TimeData`` subclass.

	See Also
	--------
	Daem
		``rp.Model`` subclass used to generate the Laplace transform for RPO
		data and translate between time- and Ea-space.

	EnergyComplex
		``rp.RateData`` subclass for storing, deconvolving, and analyzing RPO
		rate data.

	Examples
	--------
	Generating an arbitrary bare-bones thermogram containing only `t` and
	`T`::

		#import modules
		import numpy as np
		import rampedpyrox as rp

		#generate arbitrary data
		t = np.arange(1,100) #100 second experiment
		beta = 0.5 #K/second
		T = beta*t + 273.15 #K

		#create instance
		tg = rp.RpoThermogram(t,T)

	Generating a real thermogram using an RPO output .csv file and the
	``rp.RpoThermogram.from_csv`` class method::

		#import modules
		import rampedpyrox as rp

		#create path to data file
		file = 'path_to_folder_containing_data/thermogram_data.csv'

		#create instance
		tg = rp.RpoThermogram.from_csv(file,
										nt = 250,
										ppm_CO2_err = 5,
										T_err = 3)

	Manually adding some model-estimated component data as `cmpt`::

		#assuming cmpt has been generating using a Daem model
		tg.input_estimated(cmpt, 'Daem')

	Or, instead, you can input model-estimated component data directly from
	a given ``rp.Daem`` and ``rp.EnergyComplex`` instance (*i.e.* run the
	forward model)::

		tg.forward_model(daem, ec)

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

	Printing a summary of the analysis::

		print(tg.peak_info)

	**Attributes**

	cmpt : numpy.ndarray
		Array of the estimated fraction of carbon remaining in each component 
		at each timepoint. Shape [`nt` x `nPeak`].

	dcmptdt : numpy.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		in each component with respect to time at each timepoint, in 
		fraction/second. Shape [`nt` x `nPeak`].

	dcmptdT : numpy.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		in each component with respect to temperature at each timepoint, in 
		fraction/Kelvin. Shape [`nt` x `nPeak`].

	dgamdt : numpy.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		with respect to time at each timepoint, in fraction/second. Length 
		`nt`.

	dgamdT : numpy.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		with respect to temperature at each timepoint, in fraction/Kelvin.
		Length `nt`.

	dgdt : numpy.ndarray
		Array of the derivative of the true fraction of carbon remaining with
		respect to time at each timepoint, in fraction/second. Length `nt`.

	dgdT : numpy.ndarray
		Array of the derivative of the true fraction of carbon remaining with 
		respect to temperature at each timepoint, in fraction/Kelvin.
		Length `nt`.

	dof : int
		Degrees of freedom of model fit, defined as ``nt - 3*nPeak + 1``.

	dTdt : numpy.ndarray
		Array of the derivative of temperature with respect to time (*i.e.*
		the instantaneous ramp rate) at each timepoint, in Kelvin/second.
		Length `nt`.

	g : numpy.ndarray
		Array of the true fraction of carbon remaining at each timepoint.
		Length `nt`.

	g_std : numpy.ndarray
		Array of the standard deviation of `g`. Length `nt`.

	gam : numpy.ndarray
		Array of the estimated fraction of carbon remaining at each timepoint.
		Length `nt`.

	nPeak : int
		Number of peaks in estimated thermogram, **after being combined**
		(*i.e.* number of components)

	nt : int
		Number of timepoints.

	peak_info : pd.DataFrame
		Dataframe containing the forward-modeled peak isotope summary info: 

			t_max (s), \n
			T_max (K), \n
			max_rate (frac/s), \n
			max_rate (frac/K), \n
			relative area
		
		Combined peaks are treated as a single peak.

	red_chi_sq : float
		The reduced chi square metric for the model fit.

	rmse : float
		The RMSE between true and estimated thermogram.

	t : numpy.ndarray
		Array of timepoints, in seconds. Length `nt`.

	T : numpy.ndarray
		Array of temperature, in Kelvin. Length `nt`.

	T_std : numpy.ndarray
		Array of the standard deviation of `T`. Length `nt`.
	'''

	def __init__(self, t, T, g = None, g_std = 0, T_std = 0):

		#warn if T is scalar
		if isinstance(T, (int, float)):
			warnings.warn(
				'Attempting to use isothermal data for RPO run! T is a scalar'
				'value of: %.1f. Consider using an isothermal model type'
				'instead.' % T)

		super(RpoThermogram, self).__init__(
			t, 
			T, 
			g = g, 
			g_std = g_std, 
			T_std = T_std)

	#define class method for creating instance directly from .csv file
	@classmethod
	def from_csv(cls, file, nt = 250, ppm_CO2_err = 5, T_err = 3):
		'''
		Class method to directly import RPO data from a .csv file and create
		an ``rp.RpoThermogram`` class instance.

		Parameters
		----------
		file : str or pd.DataFrame
			File containing isotope data, either as a path string or a
			dataframe.

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
		format and a header row should be added with the following columns:

			date_time, \n
			T_room, \n
			P_room, \n
			CO2_raw, \n
			corr_int, \n
			corr_slope, \n
			temp, \n
			CO2_scaled, \n
			flow_rate, \n
			dTdt, \n
			fraction, \n
			ug_frac, \n
			ug_sum

		(Note that all columns besides `date_time`, `temp`, and `CO2_scaled`
		are unused.) Ensure that all rows before the start of temperature
		ramping and after the ovens have been turned off have been removed.

		When down-sampling, `t` contains the midpoints of each time bin and
		`g` and `T` contain the corresponding temp. and fraction remaining 
		at each midpoint.

		See Also
		--------
		RpoIsotopes.from_csv
			Classmethod for creating ``rp.RpoIsotopes`` instance directly from
			a .csv file.
		'''

		#extract data from file
		g, g_std, t, T = _rpo_extract_tg(file, nt, ppm_CO2_err)

		return cls(t, T, g = g, g_std = g_std, T_std = T_err)

	#define method for inputting forward-modeled data
	def forward_model(self, model, ratedata):
		'''
		Forward-models rate data for a given model and populates the
		thermogram with model-estimated data.

		Parameters
		----------
		model : rp.Model
			The ``rp.Daem`` instance used to calculate the forward model.

		ratedata : rp.RateData
			The ``rp.EnergyComplex`` instance containing the reactive 
			continuum data.

		Warnings
		--------
		UserWarning
			If time-temperature data in the ``rp.Model`` instance do not 
			match time-temperature data in the ``rp.TimeData`` instance.

		UserWarning
			If using an an isothermal model type for an RPO run.

		UserWarning
			If using a non-energy complex ratedata type for an RPO run.

		Raises
		------
		ValueError
			If `nEa` is not the same in the ``rp.Model`` instance and the 
			``rp.RateData`` instance.

		See Also
		--------
		input_estimated
			Method used for inputting model-estimated data directly.


		EnergyComplex.inverse_model
			Class for creating an ``rp.EnergyComplex`` instance and
			calculating the inverse model.
		'''

		#warn if model is not Daem
		mod_type = type(model).__name__

		if mod_type not in ['Daem']:
			warnings.warn(
				'Attempting to calculate thermogram using a model instance of'
				' type %r. Consider using rp.Daem instance instead'
				% rd_type)

		#warn if ratedata is not EnergyComplex
		rd_type = type(ratedata).__name__

		if rd_type not in ['EnergyComplex']:
			warnings.warn(
				'Attempting to calculate thermogram using a ratedata instance'
				' of type %r. Consider using rp.EnergyComplex instance'
				' instead' % rd_type)

		#raise ValueError if not the right shape
		if model.nEa != ratedata.nEa:
			raise ValueError(
				'Cannot combine model with nEa = %r and RateData with'
				' nEa = %r. Check that RateData was not created using'
				' a different model' % (model.nEa, ratedata.nEa))

		super(RpoThermogram, self).forward_model(model, ratedata)

		return

	#define method for inputting model-estimate data
	def input_estimated(self, cmpt):
		'''
		Inputs estimated thermogram into the ``tp.RpoThermogram`` instance and 
		calculates statistics.
		
		Parameters
		----------
		cmpt : array-like
			Array of the estimated fraction of carbon remaining in each 
			component at each timepoint. Shape [`nt` x `nPeak`].

		Raises
		------
		TypeError
			If `cmpt` is not array-like.

		ValueError
			If `cmpt` is not of length nt.

		See Also
		--------
		forward_model
			Method for directly inputting estimated data from a given model
			and ratedata.	
		'''

		super(RpoThermogram, self).input_estimated(cmpt)

	#define plotting method
	def plot(self, ax = None, xaxis = 'time', yaxis = 'rate'):
		'''
		Plots the true and model-estimated thermograms (including individual 
		peaks) against time or temp.

		Parameters
		----------
		ax : None or matplotlib.axis
			Axis to plot on. If `None`, automatically creates a
			``matplotlip.axis`` instance to return. Defaults to `None`.

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

		#check that axes are appropriate strings
		if xaxis not in ['time','temp']:
			raise ValueError(
				'xaxis does not accept %r. Must be either "time" or "temp"'
				%xaxis)

		elif yaxis not in ['fraction','rate']:
			raise ValueError(
				'yaxis does not accept %r. Must be either "rate" or'
				' "fraction"' %yaxis)

		#extract axis label ditionary
		rpo_labs = _plot_dicts('rpo_labs', self)
		labs = (
			rpo_labs[xaxis][yaxis][0], 
			rpo_labs[xaxis][yaxis][1])

		#check if real data exist
		if hasattr(self, 'g'):
			#extract real data dict
			rpo_rd = _plot_dicts('rpo_rd', self)
			rd = (
				rpo_rd[xaxis][yaxis][0], 
				rpo_rd[xaxis][yaxis][1])
		else:
			rd = None

		#check if modeled data exist
		if hasattr(self, 'cmpt'):
			#extract modeled data dict
			rpo_md = _plot_dicts('rpo_md', self)
			md = (
				rpo_md[xaxis][yaxis][0], 
				rpo_md[xaxis][yaxis][1], 
				rpo_md[xaxis][yaxis][2])
		else:
			md = None

		ax = super(RpoThermogram, self).plot(
			ax = ax, 
			md = md,
			labs = labs, 
			rd = rd)

		return ax

if __name__ is '__main__':

	import rampedpyrox as rp
	