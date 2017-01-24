'''
This module contains the TimeData superclass and all corresponding subclasses.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['RpoThermogram']

import matplotlib.pyplot as plt
import numpy as np
import warnings

from numpy.linalg import norm

#import exceptions
from .exceptions import(
	ArrayError,
	StringError,
	)

#import helper functions
from .core_functions import(
	assert_len,
	derivatize,
	)

from .plotting_helper import(
	_plot_dicts,
	_rem_dup_leg,
	)

from .summary_helper import(
	_calc_RPO_info,
	)

from .model_helper import(
	_calc_ghat,
	)

from .timedata_helper import(
	_rpo_extract_tg,
	)


class TimeData(object):
	'''
	Class to store time-dependent data. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, t, T, g = None, g_std = None, T_std = None):
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
			Standard deviation of `g`, with length `nt`. Defaults to `None`.

		T_std : scalar or array-like
			The standard deviation of `T`, with length `nt`, in Kelvin. 
			Defaults to `None`.
		'''

		#store time-temperature attributes
		nt = len(t)
		self.nt = nt
		self.t = assert_len(t, nt) #s
		self.T = assert_len(T, nt) #K

		if T_std is not None:

			#only store T_std if it exists (NONE for RPO, keep for future)
			self.T_std = assert_len(T_std, nt) #K

		#store time-temperature derivatives
		self.dTdt = derivatize(self.T, self.t) #K/s

		#check if g and store
		if g is not None:

			#assert that g remains between 0 and 1
			if np.max(g) > 1 or np.min(g) < 0:
				raise ArrayError(
					'g array must remain between 0 and 1 (fractional)')

			self.g = assert_len(g, nt) #fraction

			if g_std is not None:

				#only store g_std if it exists (NONE for RPO, keep for future)
				self.g_std = assert_len(g_std, nt) #fraction

			#store g derivatives
			self.dgdt = derivatize(g, self.t)
			self.dgdT = derivatize(g, self.T)

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
				' rp.TimeData instance' %(td_type, mod_type), UserWarning)


		#calculate forward-modelled g estimate, ghat
		ghat = _calc_ghat(model, ratedata)

		#populate with modelled data
		self.input_estimated(ghat)

	#define method for inputting the results from a model fit
	def input_estimated(self, ghat):
		'''
		Method to input modelled estimate data into ``rp.TimeData`` instance 
		and to calculate corresponding derivatives and statistics.

		Parameters
		----------
		ghat : array-like
			Array of estimated fraction of total carbon remaining at each 
			timestep. Length `nt`.
		'''

		#ensure type and size
		nt = self.nt
		ghat = assert_len(ghat, nt)

		#calculate derived attributes and store
		self.dghatdt = derivatize(ghat, self.t)
		self.dghatdT = derivatize(ghat, self.T)
		self.ghat = ghat

		#store RMSE if the model has true data, g
		if hasattr(self, 'g'):

			resid = norm(self.g - ghat)/nt**0.5
			self.resid = resid

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
			Tuple of modelled data, in the form  (x_data, y_data). Defaults
			to `None`.

		rd : tuple
			Tuple of real (observed) data, in the form (x_data, y_data). 
			Defaults to `None`.

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
				linewidth = 2,
				color = 'k',
				label = 'Observed Data')

			#set limits
			ax.set_xlim([0, 1.1*np.max(rd[0])])
			ax.set_ylim([0, 1.1*np.max(rd[1])])
			
		#add model-estimated data if it exists
		if md is not None:

			#plot the model-estimated total
			ax.plot(
				md[0], 
				md[1],
				linewidth = 2,
				color = 'r',
				label = 'Model-estimated data')

			#(re)set limits
			ax.set_xlim([0, 1.1*np.max(md[0])])
			ax.set_ylim([0, 1.1*np.max(md[1])])

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


class RpoThermogram(TimeData):
	__doc__='''
	Class for inputting and storing Ramped PyrOx true (observed) and estimated
	(forward-modelled) thermograms, calculating goodness of fit statistics,
	and reporting summary tables.

	Parameters
	----------
	t : array-like
		Array of time, in seconds. Length `nt`.

	T : array-like
		Array of temperature, in Kelvin. Length `nt`.

	g : None or array-like
		Array of the true fraction of carbon remaining at each timepoint,
		with length `nt`. Defaults to `None`.

	Warnings
	--------
	UserWarning
		If attempting to use isothermal data to create an ``rp.RpoThermogram``
		instance. Consider using an alternate ``rp.TimeData`` subclass (to be
		added in future versions).

	Notes
	-----
	**Important:** The inverse model used herein is highly sensitive to
	boundary effects. To avoid unnecessarily large regularizations ensure that
	inputted data are completely at baseline (ppm CO2 = 0) at the beginning
	and the end of the experiment (can use the `bl_subtract` flag to enforce
	that this is true.)

	See Also
	--------
	Daem
		``rp.Model`` subclass used to generate the distributed activation
		energy model (DAEM)transform matrix for RPO data and translate between
		time- and E-space.

	EnergyComplex
		``rp.RateData`` subclass for storing and analyzing RPO energy (rate)
		data.

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
	``rp.RpoThermogram.from_csv`` class method, and subtracting the baseline::

		#import modules
		import rampedpyrox as rp

		#create path to data file
		file = 'path_to_folder_containing_data/thermogram_data.csv'

		#create instance using baseline-subtracted CO2 data
		tg = rp.RpoThermogram.from_csv(
			file,
			bl_subtract = True,
			nt = 250) #number of down-sampled time points

	Manually adding some model-estimated fraction remaining data as `ghat`::

		#assuming ghat has been generating using a ``rp.Daem`` model
		tg.input_estimated(ghat)

	Or, instead, you can input model-estimated g data directly from a given
	``rp.Daem`` and ``rp.EnergyComplex`` instance (*i.e.* run the forward 
	model)::

		#assuming ``rp.Daem`` named daem and ``rp.EnergyComplex`` named ec
		tg.forward_model(daem, ec)

	Plotting the resulting observed and modelled thermograms (note scatter
	when plotted against temp due to short fluctuations in measured ramp rate.
	For a "smooth" plot, always plot against time, as this is the master
	variable.)::

		#import additional modules
		import matplotlib.pyplot as plt

		#create figure
		fig, ax = plt.subplots(1,2)

		#plot resulting rates against time and temp
		ax[0] = tg.plot(
			ax = ax[0], 
			xaxis = 'time', 
			yaxis = 'rate')
		
		ax[1] = tg.plot(
			ax = ax[1], 
			xaxis = 'temp', 
			yaxis = 'rate')

	Printing a summary of the observed and modelled thermograms::

		print(tg.tg_info)
		print(tg.tghat_info)

	**Attributes**


	dghatdt : numpy.ndarray
		Array of the derivative of the estimated fraction of carbon remaining
		with respect to time at each timepoint, in fraction/second. Length 
		`nt`.

	dghatdT : numpy.ndarray
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

	dTdt : numpy.ndarray
		Array of the derivative of temperature with respect to time (*i.e.*
		the instantaneous ramp rate) at each timepoint, in Kelvin/second.
		Length `nt`.

	g : numpy.ndarray
		Array of the true fraction of carbon remaining at each timepoint.
		Length `nt`.

	ghat : numpy.ndarray
		Array of the estimated fraction of carbon remaining at each timepoint.
		Length `nt`.

	nt : int
		Number of timepoints.

	resid : float
		The residual root mean square error (RMSE) between observed and
		modelled thermograms, g and ghat.

	t : numpy.ndarray
		Array of timepoints, in seconds. Length `nt`.

	T : numpy.ndarray
		Array of temperature, in Kelvin. Length `nt`.

	tg_info : pd.Series
		Series containing the observed thermogram summary info: 

			t_max (s), \n
			t_mean (s), \n
			t_std (s), \n
			T_max (K), \n
			T_mean (K), \n
			T_std (K), \n
			max_rate (frac/s), \n
			max_rate (frac/K), \n

	tghat_info : pd.Series
		Series containing the modelled thermogram summary info: 

			t_max (s), \n
			t_mean (s), \n
			t_std (s), \n
			T_max (K), \n
			T_mean (K), \n
			T_std (K), \n
			max_rate (frac/s), \n
			max_rate (frac/K), \n
	'''

	def __init__(self, t, T, g = None):

		#warn if T is scalar
		if isinstance(T, (int, float)) or len(set(T)) == 1:
			warnings.warn(
				'Attempting to use isothermal data for RPO run! T is a scalar'
				' value of: %r. Consider using an isothermal model type'
				' instead.' % T, UserWarning)

		super(RpoThermogram, self).__init__(
			t, 
			T, 
			g = g, 
			g_std = None, #force to be None for RPO
			T_std = None) #force to be None for RPO

		#if g exists, add RPO-specific summary file
		if g is not None:

			self.tg_info = _calc_RPO_info(self.t, self.T, self.g)

	#define class method for creating instance directly from .csv file
	@classmethod
	def from_csv(
			cls, 
			file, 
			bl_subtract = True, 
			nt = 250):
		'''
		Class method to directly import RPO data from a .csv file and create
		an ``rp.RpoThermogram`` class instance.

		Parameters
		----------
		file : str or pd.DataFrame
			File containing isotope data, either as a path string or a
			dataframe.

		bl_subtract : Boolean
			Tells the program whether or not to linearly subtract the baseline
			such that ppmCO2 returns to 0 at the beginning and end of the run. 
			Defaults to `True`. **To minimize boundary effects, this should 
			typically be set to `True` regardless of previous data treatment.**

		nt : int
			The number of time points to use. Defaults to 250.

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
		g, t, T = _rpo_extract_tg(
			file, 
			nt, 
			bl_subtract = bl_subtract)

		return cls(t, T, g = g)

	#define method for inputting forward-modelled data
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
			If using an an isothermal model type for an RPO run.

		UserWarning
			If using a non-energy complex ratedata type for an RPO run.

		Raises
		------
		ArrayError
			If `nE` is not the same in the ``rp.Model`` instance and the 
			``rp.RateData`` instance.

		ArrayError
			If `nt` is not the same in the ``rp.Model`` instance and the
			``rp.TimeData`` instance.

		ArrayError
			If the ``rp.RateData`` instance has no attribute `p`.

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
				% mod_type, UserWarning)

		#warn if ratedata is not EnergyComplex
		rd_type = type(ratedata).__name__

		if rd_type not in ['EnergyComplex']:
			warnings.warn(
				'Attempting to calculate thermogram using a ratedata instance'
				' of type %r. Consider using rp.EnergyComplex instance'
				' instead' % rd_type, UserWarning)

		#raise exception if no p data exist
		if not hasattr(ratedata, 'p'):
			raise ArrayError(
				'EnergyComplex has no p array!')

		#raise exception if not the right shape
		if model.nE != ratedata.nE:
			raise ArrayError(
				'Cannot combine model with nE = %r and RateData with'
				' nE = %r. Check that RateData was not created using'
				' a different model' % (model.nE, ratedata.nE))

		#raise exception if not the right shape
		if model.nt != self.nt:
			raise ArrayError(
				'Cannot combine model with nt = %r and TimeData with'
				' nt = %r. Check that the mode was not created using'
				' different time data' % (model.nt, self.nt))

		#call the superclass method
		super(RpoThermogram, self).forward_model(model, ratedata)

		return

	#define method for inputting model-estimate data
	def input_estimated(self, ghat):
		'''
		Inputs estimated thermogram into the ``rp.RpoThermogram`` instance and 
		calculates statistics.
		
		Parameters
		----------
		ghat : array-like
			Array of estimated fraction of total carbon remaining at each 
			timestep. Length `nt`.

		See Also
		--------
		forward_model
			Method for directly inputting estimated data from a given model
			and ratedata.	
		'''

		#call the superclass method
		super(RpoThermogram, self).input_estimated(ghat)

		#add RPO-specific modelled tg summary file
		self.tghat_info = _calc_RPO_info(self.t, self.T, ghat)


	#define plotting method
	def plot(self, ax = None, xaxis = 'time', yaxis = 'rate'):
		'''
		Plots the true and model-estimated thermograms against time or temp.

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
		StringError
			If `xaxis` is not 'time' or 'temp'.

		StringError
			if `yaxis` is not 'fraction' or 'rate'.
		'''

		#check that axes are appropriate strings
		if xaxis not in ['time','temp']:
			raise StringError(
				'xaxis does not accept %r. Must be either "time" or "temp"'
				%xaxis)

		elif yaxis not in ['fraction','rate']:
			raise StringError(
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

		#check if modelled data exist
		if hasattr(self, 'ghat'):
			#extract modelled data dict
			rpo_md = _plot_dicts('rpo_md', self)
			md = (
				rpo_md[xaxis][yaxis][0], 
				rpo_md[xaxis][yaxis][1])
		else:
			md = None

		ax = super(RpoThermogram, self).plot(
			ax = ax, 
			md = md,
			labs = labs, 
			rd = rd)

		return ax

if __name__ == '__main__':

	import rampedpyrox as rp
	