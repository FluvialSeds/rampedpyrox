'''
This module contains helper functions for timedata classes.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_rpo_extract_tg']

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

#import exceptions
from ..core.exceptions import(
	FileError,
	)

#define function to extract variables from .csv file
def _rpo_extract_tg(file, nt, err):
	'''
	Extracts time, temperature, and carbon remaining vectors from `all_data`
	file generated by NOSAMS RPO LabView program.

	Parameters
	----------
	file : str or pd.DataFrame
		File containing isotope data, either as a path string or a
		dataframe.

	nT : int 
		The number of time points to use.

	err : int or float
		The CO2 concentration standard deviation, in ppm. Used to 
		calculate `g_std`.

	Returns
	-------
	g : np.ndarray
		Array of the true fraction of carbon remaining at each timepoint.
		Length `nt`.

	g_std : np.ndarray
		Array of the standard deviation of `g`. Length `nt`.
	
	t : np.ndarray
		Array of timep, in seconds. Length `nt`.

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
	'''

	#check data format and raise appropriate errors
	if isinstance(file, str):
		#import as dataframe
		file = pd.read_csv(
			file,
			index_col=0,
			parse_dates=True)

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
	secs = (file.index - file.index[0]).seconds
	CO2 = file.CO2_scaled
	tot = np.sum(CO2)
	Temp = file.temp

	#calculate alpha and stdev bounds
	alpha = np.cumsum(CO2)/tot
	alpha_p = np.cumsum(CO2+err)/tot
	alpha_m = np.cumsum(CO2-err)/tot

	#generate t array
	t0 = secs[0]; tf = secs[-1]
	dt = (tf-t0)/nt

	#make downsampled points at midpoint
	t = np.linspace(t0,tf,nt+1) + dt/2 
	
	#drop last point since it's beyond tf
	t = t[:-1] 

	#generate functions to down-sample
	fT = interp1d(secs, Temp)
	fg = interp1d(secs, alpha)
	fg_p = interp1d(secs, alpha_p)
	fg_m = interp1d(secs, alpha_m)
	
	T = fT(t) + 273.15 #convert to K
	g = 1-fg(t)
	g_std = 0.5*(fg_p(t) - fg_m(t))
	g_min = fg_m(t)
	g_max = fg_p(t)

	return g, g_std, t, T

