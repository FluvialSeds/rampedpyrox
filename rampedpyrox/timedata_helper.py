'''
This module contains helper functions for timedata classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_rpo_extract_tg',
			'_bd_data_reduction']

import numpy as np
import pandas as pd
import scipy.signal as signal

from scipy.interpolate import interp1d

# #import exceptions
# from .exceptions import(
# 	FileError,
# 	)

#define function to calculate elapsed time
def _bd_calc_telapsed(
	all_index,
	t0index):
	'''
	Function to calculate elapsed time since innoculation for biodecay
	experiment.

	Parameters
	----------
	all_index : pd.DatetimeIndex
		Index of `all_data` dataframe, as pd.DatetimeIndex. No specific 
		requirements on timestep (function calculates).

	t0index : pd.Timestamp
		Innoculation time. Must be contained in `all_data` index.

	Returns
	-------
	t_elapsed : pd.Series
		Seriers containing elapsed time since innoculation, in minutes.

	Raises
	------
	ValueError
		If t0index is not contained in all_index.
	'''

	#find where timestamp = t_0 timestamp
	try:
		ind_start = np.where(all_index == t0index)[0][0]
	
	except IndexError:
		raise ValueError(
			't0 value: %r not in all_data index. Check timestamps' % t0index)

	#calculate minutes elapsed relative to t_0 timestamp
	Dt = (all_index - all_index[ind_start])
	t_elapsed = Dt / pd.Timedelta(minutes = 1)

	return t_elapsed #in minutes

#define function to convert to carbon flux in ugC min-1 L-1
def _bd_calc_ugCminL(
	input_data,
	flow_rate,
	t_elapsed,
	p_room,
	T_room,
	Vmedia,
	Fsysblk = None,
	Ctot_mano = None):
	'''
	Function to convert ppmCO2 values into ugC min-1 L-1.

	Parameters
	----------
	input_data : pd.Series
		Series containing a ppmCO2 array to be corrected, with pd.DatetimeIndex
		as index.

	flow_rate : pd.Serires
		Series containing flow rate values (in mL/min) for the experiment,
		with pd.DatetimeIndex as index.

	t_elapsed : pd.Series
		Series containing elapsed time (in min) for the experiment, with
		pd.Datetimeindex as index.

	p_room : pd.Series
		Series containing room pressure values (in kPa) for the experiment,
		with pd.DatetimeIndex as index.

	T_room : pd.Series
		Series containing room temperature values (in C) for the experiment,
		with pd.DatetimeIndex as index.

	Vmedia : pd.Series
		Series containing volume of media (in mL) remaining at each timepoint
		in the experiment, with pd.DatetimeIndex as index.

	Fsysblk : scalar or None
		Constant system blank flux (in ugC day-1) to be used for blank
		correction. If `None`, no rescaling is performed. Defaults to `None`.

	Ctot_mano : scalar or None
		The total carbon yield as determined by the sum of manometric
		measurements for each collected fraction. Used to re-scale flux values
		to ensure sum is equal to manometric sum. If `None`, no rescaling is
		performed. Defaults to `None`.

	Returns
	-------
	Cflux : pd.Series
		Seriers containing carbon flux in ugC min-1 L-1.
	'''

	#calculate multiplying scalar
	R = 8.314e3 #mL*kPa/K/mol
	Mco2 = 12.01 #g/mol
	alpha = Mco2 / (R * (T_room + 273.15)) #ug/kPa/mL/ppm
	
	#calculate carbon flux
	Cflux = input_data * alpha * p_room * flow_rate

	#calculate total photometric yield
	Dt = np.gradient(input_data.index) / pd.Timedelta(minutes = 1)
	Ctot_photo = np.sum(Cflux * Dt)

	#first, re-scale to remove system blank flux
	if Fsysblk is not None:
		#calcualte total blank C
		Ctot_sysblk = Fsysblk * t_elapsed[-1] / (60 * 24) #since F in days

		#re-scale to remove blank flux
		s1 = (Ctot_photo - Ctot_sysblk)/Ctot_photo
		Cflux = s1 * Cflux

	#then, re-scale to match manometric yield
	if Ctot_mano is not None:
		#re-scale
		s2 = Ctot_mano / Ctot_photo
		Cflux = s2 * Cflux

	# finally, re-scale for volume remaining at each time point
	Cflux = 1000 * Cflux / Vmedia

	return Cflux

#define function to correct for baseline drift
def _bd_correct_baseline(
	input_data,
	baselines):
	'''
	Function to correct biodecay ppmCO2 values for baseline drift.

	Parameters
	----------
	input_data : pd.Series
		Series containing a ppmCO2 array to be corrected, with pd.DatetimeIndex
		as index.

	baselines : pd.Series
		Series containing baseline values, with pd.DatetimeIndex as index.

	Returns
	-------
	bl_corr : pd.Series
		Seriers containing baseline corrected ppmCO2 array.

	Raises
	------
	ValueError
		If `baselines` index values are not in `input_data` index.

	Notes
	-----
	Function forces ppmCO2 at t0 to be zero. Assumes no drift after final
	baseline check.
	'''

	#extract t0 index
	t0_index = baselines.index[0]

	#only retain timepoints with baseline values
	baselines.dropna(inplace = True)
	
	#calcualte indices for baseline timepoints
	bl_inds = baselines.index

	#check indices
	if not all([a in input_data.index for a in bl_inds]):
		raise ValueError(
			'baseline timestamps not contained in all_data index.' \
			' Check timestamps!')

	#calculate indices for timepoints following bl_inds (zero timepoints)
	bl_locs = np.array([input_data.index.get_loc(a) for a in bl_inds])
	z_locs = bl_locs + 1
	z_inds = input_data.iloc[z_locs].index

	#calculate baseline CO2 array
	CO2_bl = pd.Series(index = input_data.index)
	
	#set CO2 at bl_inds to be inputted bl values, at z_inds to be zero
	CO2_bl[bl_inds] = baselines
	CO2_bl[z_inds] = 0
	
	#set baseline CO2 equal to total CO2 at intitial point and at t0
	CO2_bl[0] = input_data[0]
	CO2_bl[t0_index] = input_data[t0_index]

	#linearly interpolate
	CO2_bl.interpolate(
		method = 'linear',
		inplace = True
		)

	#subtract baseline ppmCO2 values to correct data
	bl_corr = input_data - CO2_bl

	return bl_corr

#define function to correct for headspace averaging and volume changes
def _bd_correct_headspace(
	input_data,
	flow_rate,
	samples,
	Vmedia0 = 2000,
	Vhs0 = 4000):
	'''
	Function to correct biodecay ppmCO2 values for headspace averaging and
	volume changes due to liquid sub-sampling.

	Parameters
	----------
	input_data : pd.Series
		Series containing a ppmCO2 array to be corrected, with pd.DatetimeIndex
		as index.

	flow_rate : pd.Serires
		Series containing flow rate values (in mL/min) for the experiment,
		with pd.DatetimeIndex as index.

	samples : pd.Series
		Series containing sub-sample volumnes (in mL), with pd.DatetimeIndex 
		as index.

	Vmedia0 : scalar
		Initial volume of media for experiment, in mL. Defaults to 2000.

	Vhs0 : scalar
		Initial headspace volume for experiment, in mL. Defaults to 4000.

	Returns
	-------
	hs_corr : pd.Series
		Seriers containing headspace- and volume-corrected ppmCO2 array.

	Vmedia : pd.Series
		Series of liquid media remaining at each time point.

	Raises
	------
	ValueError
		If `samples` index values are not in `input_data` index.
	'''

	#only retain timepoints where subsamples were taken
	samples.dropna(inplace = True)
	
	#calcualte indices for sampling timepoints
	sam_inds = samples.index

	#check indices
	if not all([a in input_data.index for a in sam_inds]):
		raise ValueError(
			'liq_sample timestamps not contained in all_data index.' \
			' Check timestamps!')

	#make array of volume sampled
	Vsampled = pd.Series(
		index = input_data.index
		)
	
	Vsampled[0] = 0 #initialize to zero
	Vsampled[sam_inds] = samples.cumsum()
	Vsampled = Vsampled.fillna(
		method = 'ffill'
		)

	#make arrays of volume remaining and headspace volume
	Vmedia = Vmedia0 - Vsampled
	Vhs = Vhs0 + Vsampled

	#correct ppmCO2 to acount for nonzero residence time in headspace
	# first, calculate time steps (in mins) to account for ragged arrays
	Dt = np.gradient(input_data.index) / pd.Timedelta(minutes = 1)

	# then, find first derivative of input data and filter to remove noise
	dCdt_vals = np.gradient(input_data) / Dt
	dCdt = pd.Series(
		dCdt_vals, 
		index = input_data.index)

	dCdt = _bd_rolling(dCdt, 
		window = 100,
		center = True,
		calc = 'mean')

	# finally, correct for headspace averaging using first-order, linear ODE
	hs_corr = (Vhs/flow_rate) * dCdt + input_data

	return hs_corr, Vmedia

#define function to input biodecay data and perform reduction calculations
def _bd_data_reduction(
	all_file,
	sam_file,
	mins_before_zero = 30,
	Vmedia0 = 2000, #intial media volumne, in mL
	Vhs0 = 4000, #initial headspace volumne, in mL
	Fsysblk = 10, #system blank flux, in ugC/day
	Ctot_mano = None): #total manometric C yield
	'''
	Parameters
	----------

	Returns
	-------

	Raises
	------

	Notes
	-----
	'''

	#check all_file data format and raise appropriate errors
	
	#all_file format
	if isinstance(all_file, str):
		#import as dataframe
		all_data = pd.read_csv(
			all_file,
			index_col = 0,
			parse_dates = True)

	elif isinstance(all_file, pd.DataFrame):
		#rename to all_data
		all_data = all_file

	else:
		raise FileError(
			'all_file must be pd.DataFrame instance or path string')

	#all_file data
	
	#list of necessary columns
	ad_cols = ['temp','p_room','CO2_scaled','flow_rate']

	if not isinstance(all_data.index, pd.DatetimeIndex):
		raise FileError(
			'all_file index (first column of csv file) must be' \
			' in date_time format (pd.DatetimeIndex instance)'
			)

	elif not all([a in all_data.columns for a in ad_cols]):
		raise FileError(
			'all_file must contain columns: %r' % ad_cols
			)

	#check sam_file data format and raise appropriate errors

	#sam_file format
	if isinstance(sam_file, str):
		#import as dataframe
		sam_data = pd.read_csv(
			sam_file,
			index_col = 0,
			parse_dates = True)

	elif isinstance(sam_file, pd.DataFrame):
		#rename to sam_data
		sam_data = sam_file

	else:
		raise FileError(
			'sam_file must be pd.DataFrame instance or path string')


	#sam_file data

	#list of necessary columns
	sd_cols = ['CO2_bl','liq_sample','cell_ct']

	if not isinstance(sam_data.index, pd.DatetimeIndex):
		raise FileError(
			'sam_file index (first column of csv file) must be' \
			' in date_time format (pd.DatetimeIndex instance)'
			)

	elif not all([a in sam_data.columns for a in sd_cols]):
		raise FileError(
			'sam_file must contain columns: %r' % ad_cols
			)

	#---------------------------------------------------#
	# 1) REMOVE SPIKES AND CALCULATE t ARRAY IN MINUTES #
	#---------------------------------------------------#

	#because seconds might have gotten dropped during import, downsample to
	# once per minute by averaging over all points in a given minute
	all_data = all_data.resample(
		'1T'
		).median()
	
	all_data.interpolate(
		method = 'linear', 
		inplace = True
		)

	#calculate time elapsed in minutes
	all_data['t_elapsed'] = _bd_calc_telapsed(
		all_data.index,
		sam_data.index[0]
		)

	#drop everything before mins_before_zero
	all_data = all_data[all_data['t_elapsed'] > -(mins_before_zero+1)]

	#calculte spike-removed CO2
	all_data['CO2_filt'] = _bd_rolling(
		all_data['CO2_scaled'],
		window = 10,
		center = True,
		calc = 'median'
		)

	#--------------------------#
	# 2) REMOVE BASELINE DRIFT #
	#--------------------------#

	all_data['CO2_blcorr'] = _bd_correct_baseline(
		all_data['CO2_filt'],
		sam_data['CO2_bl']
		)

	#------------------------------------#
	# 3) CORRECT FOR HEADSPACE AVERAGING #
	#------------------------------------#

	all_data['CO2_nohs'], Vmedia = _bd_correct_headspace(
		all_data['CO2_blcorr'],
		all_data['flow_rate'],
		sam_data['liq_sample'],
		Vmedia0 = Vmedia0,
		Vhs0 = Vhs0
		)

	#-----------------------------------------------------------#
	# 4) CONVERT TO ugC/min, RESCALE, AND SUBTRACT SYSTEM BLANK #
	#-----------------------------------------------------------#

	all_data['ugC_minL'] = _bd_calc_ugCminL(
		all_data['CO2_nohs'],
		all_data['flow_rate'],
		all_data['t_elapsed'],
		all_data['p_room'],
		all_data['temp'],
		Vmedia,
		Fsysblk = Fsysblk,
		Ctot_mano = Ctot_mano
		)

	return all_data

#define function to calculate rolling values
def _bd_rolling(
	input_data,
	window = 10,
	center = True,
	calc = 'median'):
	'''
	Function to calculate rolling values (means, medians, etc.) for biodecay
	experiment.

	Parameters
	----------
	input_data : pd.Series
		Series containing a ppmCO2 array to be rolled.

	window : int
		Number of timesteps to integrate over for rolling values. Devaults to
		10.

	center : boolean
		Tells the funciton whether or not to center the rolling values.
		Defaults to `True`.

	calc : str
		Type of rolling value to take. Either 'mean', 'median', or 'sum'.
		Defaults to 'median'.

	Returns
	-------
	rolled : pd.Series
		Seriers containing rolled ppmCO2 array.

	Raises
	------
	ValueError
		If `calc` is not one of 'mean', 'median', or 'sum'.

	Notes
	-----
	`window` is the window width *not* the time width. If `input_data`
	contains uneven timesteps, then `window` will not be a constant size in
	time. (termed "ragged" in pandas notation.)
	'''

	#raise errors
	if calc not in ['mean','median','sum']:
		raise ValueError(
			'calc: %r not recognized. Must be mean, median, or sum' % calc)

	#calculate rolling
	roll = input_data.rolling(
		window = window,
		center = center,
		)

	#execute rolling
	rolled = getattr(roll, calc)()

	#forward- and back-fill nans
	rolled = rolled.fillna(
		method = 'ffill'
		)

	rolled = rolled.fillna(
		method = 'bfill'
		)

	return rolled

#define function to extract variables from .csv file
def _rpo_extract_tg(file, nt, bl_subtract = True):
	'''
	Extracts time, temperature, and carbon remaining vectors from `all_data`
	file generated by NOSAMS RPO LabView program.

	Parameters
	----------
	file : str or pd.DataFrame
		File containing thermogram data, either as a path string or a
		dataframe.

	nt : int 
		The number of time points to use.

	bl_subtract : Boolean
		Tells the program whether or not to linearly subtract the baseline
		such that ppm CO2 returns to 0 at the beginning and end of the run. 
		Treats baseline as linear from the average of the first 100 points in
		CO2 to the average of the last 100 points in CO2. Defaults to`True`.

	Returns
	-------
	g : np.ndarray
		Array of the true fraction of carbon remaining at each timepoint.
		Length `nt`.
	
	t : np.ndarray
		Array of time, in seconds. Length `nt`.

	T : np.ndarray
		Array of temperature, in Kelvin. Length `nt`.

	Raises
	------
	FileError
		If `file` is not str or ``pd.DataFrame`` instance.
	
	FileError
		If index of `file` is not ``pd.DatetimeIndex`` instance.

	FileError
		If `file` does not contain "CO2_scaled" and "temp" columns.

	Notes
	-----
	Noisy data, especially at the beginning of the run, could lead to `g`
	values slightly outside of the (0, 1) range (*i.e.* noisy ppm CO2 less than
	zero could lead to slightly negative `g`). This method removes this
	possibility by enforcing that all values of `g` are within (0, 1).

	'''

	#check data format and raise appropriate errors
	if isinstance(file, str):
		#import as dataframe
		file = pd.read_csv(
			file,
			index_col = 0,
			parse_dates = True)

	elif not isinstance(file, pd.DataFrame):
		raise FileError(
			'file must be pd.DataFrame instance or path string')

	if 'CO2_scaled' and 'temp' not in file.columns:
		raise FileError(
			'file must have "CO2_scaled" and "temp" columns')

	elif not isinstance(file.index, pd.DatetimeIndex):
		raise FileError(
			'file index must be pd.DatetimeIndex instance')

	#extract necessary data
	secs_m = (file.index - file.index[0]).seconds.values
	CO2_m = file.CO2_scaled
	T_m = file.temp

	#before continuing, thermogram must be linearly interpolated to deal
	# with missing data (e.g. if data deleted during data clean-up).
	# Version: 0.1.3., bug noticed by Cristina Subt (SFU).

	#make array of 1-second deltas
	secs = np.arange(0,np.max(secs_m))

	#make CO2 function and interpolate
	fsm = interp1d(secs_m, CO2_m)
	CO2 = fsm(secs)

	#make Temp function and interpolate
	ftm = interp1d(secs_m, T_m)
	Temp = ftm(secs)

	#stop Version 0.1.3. bug fix here.

	#linearly subtract baseline if required
	if bl_subtract is True:
		#calculate initial and final bl
		bl0 = np.average(CO2[:100])
		blf = np.average(CO2[-100:])

		#subtract linear baseline
		npt = len(CO2)
		bl = bl0 + (blf - bl0)*np.arange(0, npt)/npt

		CO2 = CO2 - bl

	#calculate total CO2
	tot = np.sum(CO2)

	#calculate alpha
	alpha = np.cumsum(CO2)/tot

	#assert that alpha remains between 0 and 1 (noisy data at the beginning 
	# and end of the run could lead to alpha *slightly* outside fo this range)
	alpha[alpha > 1.0] = 1.0
	alpha[alpha < 0.0] = 0.0

	#generate t array
	t0 = secs[0]; tf = secs[-1]
	dt = (tf-t0)/nt

	#make downsampled points at midpoint
	t = np.linspace(t0, tf, nt + 1) + dt/2 
	
	#drop last point since it's beyond tf
	t = t[:-1] 

	#generate functions to down-sample
	fT = interp1d(secs, Temp)
	fg = interp1d(secs, alpha)
	
	#create downsampled arrays
	T = fT(t) + 273.15 #convert to K
	g = 1-fg(t)

	return g, t, T
