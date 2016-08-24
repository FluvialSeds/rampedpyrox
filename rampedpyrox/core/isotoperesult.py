'''
This module contains the IsotopeResult class for calculating the isotope
composition of individual Ea peaks within a sample, as well as supporting
functions.

* TODO: Everything...
'''

import numpy as np
import pandas as pd

def _blank_correct():
	'''
	Performs blank correction (NOSAMS RPO instrument) on raw isotope values.
	'''

def _calc_cont_ptf():
	'''
	Calculates the contribution of each peak to each fraction.
	'''

	return cont_ptf

def _d13C_to_13R(d13C, d13C_std):
	'''
	Converts d13C values to 13R values using VPDB standard.

	Args:
		d13C (np.ndarray): Inputted d13C values.
		d13C_std (np.ndarray): Inputted d13C stdev.

	Returns:
		R13 (np.ndarray): d13C values converted to 13C ratios.
		R13_std (np.ndarray): d13C stdev. values converted to 13C ratios.
	'''

	Rpdb = 0.011237 #13C/12C ratio VPDB

	R13 = ((d13C/1000)+1)*Rpdb
	R13_std = Rpdb*d13C_std/1000

	return R13, R13_std

def _extract_isotopes(sum_data):
	'''
	Extracts isotope data from the "sum_data" file.

	Args:
		sum_data (str or pd.DataFrame): File containing isotope data,
			either as a path string or pandas.DataFrame object.

	Returns:
		t (np.ndarray): 2d array of times, 1st column is t0, 2nd column is tf
		R13 (np.ndarray): 2d array of 13R values, 1st column is mean, 2nd
			column is stdev.
		Fm (np.ndarray): 2d array of Fm values, 1st column is mean, 2nd
			column is stdev.

	Raises:
		ValueError: If `sum_data` is not str or pd.DataFrame.
		ValueError: If `sum_data` does not contain "d13C", "d13C_std", "Fm",
			"Fm_std", and "fraction" columns.
		ValueError: If index is not `DatetimeIndex`.
		ValueError: If first two rows are not fractions "-1" and "0"
	'''

	#import sum_data as a pd.DataFrame if inputted as a string path and check
	#that it is in the right format
	if isinstance(sum_data,str):
		sum_data = pd.DataFrame.from_csv(sum_data)
	elif not isinstance(sum_data,pd.DataFrame):
		raise ValueError('sum_data must be pd.DataFrame or path string')

	if 'fraction' and 'd13C' and 'd13C_std' and 'Fm' and 'Fm_std' \
		not in sum_data.columns:
		raise ValueError('sum_data must have "fraction", "d13C", "d13C_std",'\
			' "Fm", and "Fm_std" columns')

	if not isinstance(sum_data.index,pd.DatetimeIndex):
		raise ValueError('sum_data index must be DatetimeIndex')

	if sum_data.fraction[0] != -1 or sum_data.fraction[1] != 0:
		raise ValueError('First two rows must be fractions "-1" and "0"')

	#extract time data
	secs = (sum_data.index - sum_data.index[0]).seconds
	t0 = secs[1:-1]
	tf = secs[2:]
	t = np.column_stack((t0,tf))

	#extract d13C data and convert to ratio
	d13C_mean = sum_data.d13C[2:]
	d13C_std = sum_data.d13C_std[2:]
	R13_mean,R13_std = _d13C_to_13R(d13C_mean, d13C_std)
	R13 = np.column_stack((R13_mean,R13_std))

	#extract Fm data
	Fm_mean = sum_data.Fm[2:]
	Fm_std = sum_data.Fm[2:]
	Fm = np.column_stack((Fm_mean,Fm_std))
	
	return t, R13, Fm


class IsotopeResult(object):
	'''
	Class for performing isotope deconvolution
	'''

	def __init__(self, sum_data, mod_tg, blank_correct=True, DEa=None):

		#extract isotopes and time
		t,R13,Fm = _extract_isotopes(sum_data)

		#define public parameters
		self.t = t
		self.R13_frac = R13
		self.Fm_frac = Fm

		#perform regression

		#


	def summary():
		'''
		Prints a summary of the IsotopeResult.
		'''