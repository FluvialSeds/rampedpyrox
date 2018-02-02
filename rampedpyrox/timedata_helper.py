'''
This module contains helper functions for timedata classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_bd_calc_BGE',
			'bd_calc_telapsed',
			'_bd_calc_ugCminL',
			'_bd_correct_baseline',
			'_bd_correct_headspace',
			'_bd_data_reduction',
			'_bd_plot_bge',
			'_bd_rolling',
			'_rpo_extract_tg',
			]

import numpy as np
import pandas as pd
import scipy.signal as signal

from scipy.interpolate import interp1d

# #import exceptions
# from .exceptions import(
# 	FileError,
# 	)

#define function to calculate bacterial growth efficiency (BGE)
def _bd_calc_BGE(
	Cflux,
	cell_counts,
	Cflux_err = None,
	cell_counts_err = None,
	alpha = [50, 1]):
	'''
	Function to calculate bacterial growth efficiency (BGE) over the course of
	a biodecay experiment.

	Parameters
	----------
	Cflux : pd.Series
		Series containing carbon flux in ugC min-1 L-1, with pd.DatetimeIndex 
		as index.

	cell_counts : pd.Series
		Series containing cell counts at each sampling point (in units of
		cells per mL), with pd.DatetimeIndex as index.

	Cflux_err : None or pd.Series
		Series containing Cflux uncertainty to be used to calculate BGE
		uncertainty. If `None`, Cflux is assumed to be known perfectly.
		Defaults to `None`.

	cell_counts_err : None, scalar, or pd.Series
		Series containing cell count uncertainty to be used to calculate BGE
		uncertainty. If `None`, cell counts are assumed to be known perfectly.
		If scalar, assumed to be fractional uncertainty and applied to all
		values (i.e. a value of 0.01 means all measurements have 1 % relative
		uncertainty). Defaults to `None`.

	alpha : list
		List of mass of carbon per cell and associated uncertainty 
		(+/- 1 sigma), in femtograms. Defaults to `[50, 1]`.

	Returns
	-------
	bge : pd.Series
		Series containing calculated BGE values, reported at the final
		timepoint for a given value.

	Raises
	------
	ValueError
		If `cell_counts` timestamps are not contained in `Cflux` index.

	References
	----------
	PER CELL CARBON MASS REFERENCE? GET FROM NAGISSA!

	'''

	#only retain timepoints with cell count values
	cell_counts.dropna(inplace = True)
	
	#calcualte indices for cell count timepoints
	cc_inds = cell_counts.index

	#check indices
	if not all([a in Cflux.index for a in cc_inds]):
		raise ValueError(
			'cell_count timestamps not contained in Cflux index.' \
			' Check timestamps!')

	#if errors are NoneType, make arrays of zeros instead
	if Cflux_err is None:
		Cflux_err = pd.Series(
			0,
			index = Cflux.index
			)

	if cell_counts_err is None:
		cell_counts_err = pd.Series(
			0,
			index = cell_counts.index
			)

	#or if scalar, broadcast out to be a Series
	elif type(cell_counts_err) in [int, float]:
		cell_counts_err = cell_counts_err * cell_counts

	#calculate microbial biomass C in ug L-1 at each time point
	cellC = alpha[0] * cell_counts * 1e-6
	cellC_err = 1e-6 * ((cell_counts * alpha[1])**2 + \
		(alpha[0] * cell_counts_err)**2 )**0.5

	#calculate difference and drop first entry (will be NaN)
	DcellC = cellC.diff()[1:]
	DcellC_err = (cellC_err**2).rolling(2).sum()**0.5

	#calculate timestep in minutes
	Dt = np.gradient(Cflux.index) / pd.Timedelta(minutes = 1)

	#integrate Cflux curve
	cumC = Dt*Cflux.cumsum() #total ug per L
	cumC_err = (np.cumsum((Cflux_err * Dt)**2))**0.5

	#determine change in CO2 mass between cell count points
	cumC_cc = cumC[cc_inds]
	cumC_cc_err = cumC_err[cc_inds]
	DcumC_cc = cumC_cc.diff()
	DcumC_cc_err = ((cumC_cc_err**2).rolling(2).sum()**0.5)

	x = DcumC_cc / DcellC
	x_err = ((DcumC_cc_err / DcellC)**2 + \
		(DcumC_cc * DcellC_err / (DcellC**2) )**2)**0.5

	#calculate BGE and error
	bge = 1 / (1 + x)
	bge_err = x_err * (1 + x)**-2

	return bge, bge_err

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
	input_data_err,
	flow_rate,
	t_elapsed,
	p_room,
	T_room,
	Vmedia,
	Fsysblk = [10, 1],
	Ctot_mano = None,
	mano_error = 0.01):
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

	Fsysblk : list or None
		Constant flux of system blank carbon, as reported in Beaupre et al.,
		(2016), in units of ugC day-1. First value is the mean and second
		value is the +/- 1 sigma uncertainty. Defaults to `[10, 1]`. If `None`,
		no correction is performed.

	Ctot_mano : scalar or None
		The total carbon yield as determined by the sum of manometric
		measurements for each collected fraction. Used to re-scale flux values
		to ensure sum is equal to manometric sum. If `None`, no rescaling is
		performed. Defaults to `None`.

	mano_erro : scalar
		One-sigma relative uncertainty of the manometer, to be used when
		scaling photometric yields to manometric ones. Reported in fraction
		of total mass (i.e. a value of 0.01 means 1% relative uncertainty).
		Defualts to `0.01`.

	Returns
	-------
	Cflux : pd.Series
		Seriers containing carbon flux in ugC min-1 L-1.

	Cflux_err : pd.Series
		Series containing +/- 1sigma uncertaint of carbon flux in 
		ugC min-1 L-1.
	'''

	#calculate multiplying scalar
	R = 8.314e3 #mL*kPa/K/mol
	Mco2 = 12.01 #g/mol
	alpha = Mco2 / (R * (T_room + 273.15)) #ug/kPa/mL/ppm
	
	#calculate carbon flux and associated uncertainty
	Cflux = input_data * alpha * p_room * flow_rate
	Cflux_err = input_data_err * alpha * p_room * flow_rate

	#calculate total photometric yield
	Dt = np.gradient(input_data.index) / pd.Timedelta(minutes = 1)
	Ctot_photo = np.sum(Cflux * Dt)
	Ctot_photo_err = (np.sum((Cflux_err * Dt)**2))**0.5

	#first, re-scale to remove system blank flux
	if Fsysblk is not None:
		#calcualte total blank C
		Ctot_sysblk = Fsysblk[0] * t_elapsed[-1] / (60 * 24) #since F in days
		Ctot_sysblk_err = Fsysblk[1] * t_elapsed[-1] / (60 * 24)

		#re-scale to remove blank flux
		s1 = (Ctot_photo - Ctot_sysblk)/Ctot_photo
		s1_err = ((Ctot_sysblk_err / Ctot_photo)**2 + \
			(Ctot_sysblk * Ctot_photo_err / (Ctot_photo**2) )**2)**0.5

		#caclualted updated C flux and uncertainty
		Cflux = s1 * Cflux
		Cflux_err = ((s1_err)**2 + (Cflux_err)**2)**0.5

	#then, re-scale to match manometric yield
	if Ctot_mano is not None:
		#re-scale
		s2 = Ctot_mano / Ctot_photo

		#calculate uncertainty
		Ctot_mano_err = Ctot_mano * mano_error
		s2_err = ((Ctot_mano_err / Ctot_photo)**2 + \
			(Ctot_mano * Ctot_photo_err / (Ctot_photo**2) )**2)**0.5

		Cflux = s2 * Cflux
		Cflux_err = ((s2_err)**2 + (Cflux_err)**2)**0.5

	# finally, re-scale for volume remaining at each time point
	Cflux = 1000 * Cflux / Vmedia
	Cflux_err = 1000 * Cflux_err / Vmedia

	return Cflux, Cflux_err

#define function to correct for baseline drift
def _bd_correct_baseline(
	input_data,
	baselines,
	IRGA_error = 1):
	'''
	Function to correct biodecay ppmCO2 values for baseline drift.

	Parameters
	----------
	input_data : pd.Series
		Series containing a ppmCO2 array to be corrected, with pd.DatetimeIndex
		as index.

	baselines : pd.Series
		Series containing baseline values, with pd.DatetimeIndex as index.

	IRGA_error : scalar
		One-sigma uncertainty of the infrared gas analyzer, in ppm.
		Defaults to `1`.

	Returns
	-------
	bl_corr : pd.Series
		Seriers containing baseline corrected ppmCO2 array.

	bl_corr_err : pd.Series
		Series containing 1sigma uncertainty in baseline corrected ppmCO2
		array.

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

	#assume constant IRGA error for both baseline and measurement
	# total error is thus sqrt(2*(IRGA error)^2) for all time points
	err_val = (2*IRGA_error**2)**0.5
	
	bl_corr_err = pd.Series(
		err_val,
		bl_corr.index
		)

	return bl_corr, bl_corr_err

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
	Vmedia0 = 2000,
	Vhs0 = 4000,
	Fsysblk = [10, 1],
	downsampled_dt = None,
	Ctot_mano = None,
	IRGA_error = 1.0,
	mano_error = 0.01):
	'''
	Inputs `all_data` file file from IsoCaRB instrument at Harvard and
	performs all necessary data corrections and checks.

	Parameters
	----------
	all_file : str or pd.DataFrame
		File containing timeseries data, either as a path string or a
		dataframe.

	sam_file : str or pd.DataFrame
		File containing sampling data (baseline checks, liquid subsampling
		times and volumes, cell counts), either as a path string or a
		dataframe. First row must contain innoculation time (t0) timestamp.

	mins_before_zero : scalar
		Number of minutes before t0 to retain in final data. Defaults to `30`.

	Vmedia0 : scalar
		Initial volume of media used in experiment, in mL. Defaults to `2000`.

	Vhs0 : scalar
		Initial headspace volume for experiment, in mL. Defaults to `4000`.

	Fsysblk : list
		Constant flux of system blank carbon, as reported in Beaupre et al.,
		(2016), in units of ugC day-1. First value is the mean and second
		value is the +/- 1 sigma uncertainty. Defaults to `[10, 1]`.

	downsampled_dt : scalar or None:
		Timestep to be used in final, downsampled data (in minutes). If 
		`None`, no downsampling is performed and returned `all_data` will 
		have 1-minutes timesteps.

	Ctot_mano : scalar or None
		The total carbon yield as determined by the sum of manometric
		measurements for each collected fraction. Used to re-scale flux values
		to ensure sum is equal to manometric sum. If `None`, no rescaling is
		performed. Defaults to `None`.

	IRGA_error : scalar
		One-sigma uncertainty of the infrared gas analyzer, in ppm.
		Defaults to `1`.

	mano_erro : scalar
		One-sigma relative uncertainty of the manometer, to be used when
		scaling photometric yields to manometric ones. Reported in fraction
		of total mass (i.e. a value of 0.01 means 1% relative uncertainty).
		Defualts to `0.01`.


	Returns
	-------
	all_data : pd.DataFrame
		Dataframe containing all reduced and corrected data. If inputted
		`downsampled_dt` parameter is not `None`, then `all_data` will be
		downsampled.

	sam_data : pd.DataFrame
		Dataframe containing all sampling data (baseline checks, liquid 
		subsampling times and volumes, cell counts).

	Raises
	------
	FileError
		If `all_file` is not str or ``pd.DataFrame`` instance.

	FileError
		If `sam_file` is not str or ``pd.DataFrame`` instance.

	FileError
		If index of `all_file` is not ``pd.DatetimeIndex`` instance.

	FileError
		If index of `sam_file` is not ``pd.DatetimeIndex`` instance.

	FileError
		If `all_file` does not contain "temp", "p_room", "CO2_scaled", and 
		"flow_rate" columns.

	FileError
		If `sam_file` does not contain "CO2_bl" and "liq_sample" columns.

	Notes
	-----
	This function automates all data reduction steps that were originally
	performed using Beaupre excel spreadsheets. Function was written following
	the steps in Beaupre "Steps for IsoCaRB Data Reduction" lab document.

	Error is propagated using the following assumptions:
	[1] IRGA has constant, user-defined error (typically +/-1 ppm 1sigma)
	[2] Headspace volume, gas flow rate, room temp., and room pressure are
	known exactly and have no uncertainty.
	[3] When scaling fluxes to manometric measurements, manometer has user-
	defined relative uncertainty, typically 1%.
	[4] System blank flux has constant, user-defined uncertainty (typically
	+/- 1 ugC/day 1 sigma, as defined in Beaupre et al., 2016)
	[5] Blank, baseline, and measured ppmCO2 uncertainties are independent.
	This is not strictly true, but a fine approximation.

	References
	----------
	[1] S. Beaupre et al. (2016) IsoCaRB: A novel bioreactor system to
	characterize the lability and natural carbon isotopic (14C, 13C)
	signatures of microbially respired organic matter. *L&O Methods*, **14**, 
	668-681.

	[2] N. Mahmoudi et al. (2017) Sequential bioavailability of sedimentary 
	organic matter to heterotrophic bacteria. *Environmental Microbiology*,
	**19**, 2629-2644.

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
	sd_cols = ['CO2_bl','liq_sample']

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
	# 1) CALCULATE t ARRAY IN MINUTES AND REMOVE SPIKES #
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

	all_data['CO2_blcorr'], all_data['CO2_err'] = _bd_correct_baseline(
		all_data['CO2_filt'],
		sam_data['CO2_bl'],
		IRGA_error = IRGA_error
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

	all_data['ugC_minL'], all_data['Cflux_err'] = _bd_calc_ugCminL(
		all_data['CO2_nohs'],
		all_data['CO2_err'],
		all_data['flow_rate'],
		all_data['t_elapsed'],
		all_data['p_room'],
		all_data['temp'],
		Vmedia,
		Fsysblk = Fsysblk,
		Ctot_mano = Ctot_mano,
		mano_error = mano_error
		)

	#---------------#
	# 5) DOWNSAMPLE #
	#---------------#

	if downsampled_dt is not None:
		dt = str(downsampled_dt) + 'T'

		all_data = all_data.resample(
			dt
			).median()

	return all_data, sam_data

#define function to plot carbon flux overlaid by BGE
def _bd_plot_bge(
	t_elapsed,
	bge,
	bge_err = None,
	ax = None,
	ymin = 0.0,
	ymax = 1.0):
	'''
	Function to plot the carbon flux (in ugC min-1 L-1) overlaid by bacterial
	growth efficiency for each time bin.

	Parameters
	----------
	t_elapsed : pd.Series
		Series containing the time elapsed (in minutes), with pd.DatetimeIndex
		as index.

	bge : pd.Series
		Series containing calculated BGE values, reported at the final
		timepoint for a given value.

	bge_err : None or pd.Series
		Series containing uncertainties for BGE values, reported at the final
		timepoint for a given value. If `None`, no uncertainty is plotted.
		Defaults to `None`.

	ax : None or matplotlib.axis
		Axis to plot BGE data on. If `None`, automatically creates a
		``matplotlip.axis`` instance to return. Defaults to `None`.

	ymin : float
		Minimum y value for BGE axis. Defaults to `0.0`.

	ymax : float
		Maximum y value for BGE axis. Defaults to `1.0`.

	Returns
	-------
	ax : matplotlib.axis
		Axis containing BGE data
	'''
	#create axis if necessary and label
	if ax is None:
		_, ax = plt.subplots(1, 1)

	#find t_elapsed values for each entry in bge
	bge_inds = bge.index
	bge_times = t_elapsed[bge_inds]

	#loop through each time range and plot BGE
	for i, ind in enumerate(bge_inds[1:]):
		#find bounding time points
		t0 = t_elapsed[bge_inds[i]]
		tf = t_elapsed[ind]
		b = bge[i+1]

		#plot results
		ax.plot(
			[t0, tf],
			[b, b],
			c = 'k',
			linewidth = 2
			)

		#include uncertainty as a shaded box
		if bge_err is not None:

			berr = bge_err[i+1]

			ax.fill_between(
				[t0, tf],
				b - berr,
				b + berr,
				alpha = 0.5,
				color = 'k',
				linewidth = 0
				)

	#set limits and label
	ax.set_ylim([ymin, ymax])
	ax.set_ylabel('Bacterial Growth Efficiency (BGE)')

	return ax

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
