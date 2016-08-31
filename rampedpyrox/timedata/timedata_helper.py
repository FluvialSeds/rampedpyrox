'''
This module contains helper functions for timedata classes.
'''

import numpy as np

#function to ensure all arrays are of length nt
def _assert_lent(array, nt):
	'''
	Helper function to ensure all arrays are of length nt.

	Parameters
	----------
	array : scalar, array-like, or None
		Array to ensure is length nt. If scalar, returns array of repeated
		scalar value. If None, returns None.

	nt : int
		Length of array to ensure

	Returns
	-------
	array : np.ndarray or None
		Array of length nt, or None.

	Raises
	------
	TypeError
		If `array` is not scalar, array-like, or None.

	TypeError
		If `nt` is not int.

	ValueError
		If `array` is array-like but not of length nt.

	'''

	#check nt is int
	if not isinstance(nt, int):
		raise TypeError('nt must be int')

	#ensure array is the right length or is None
	if array is None:
		return array

	elif isinstance(array, (int, float)):
		return array*np.ones(nt)

	elif isinstance(array, (list, np.ndarray)):
		if len(array) != nt:
			raise ValueError('array must have length nt')

		return np.array(array)
	
	else:
		raise TypeError('array bust be scalar, array-like, or None')

#function to numerically derivatize an array wrt another array
def _derivatize(num, denom):
	'''
	Helper function to calculate derivatives.

	Parameters
	----------
	num : scalar, array-like, or None
		Numerator of derivative function, length nt.

	denom : array-like, or None
		Denominator of derivative function, length nt. Returns `undefined` if
		`denom` is not continuously increasing.

	Returns
	-------
	dndd : np.ndarray or 'undefined'
		Derivative of `num` wrt `denom`. Returns 'undefined' if the
		derivative could not be calculated. Returns an array of zeros of
		length nt if `num` is a scalar.

	Raises
	------
	TypeError
		If `num` is not scalar, array-like, or None.
	
	TypeError
		If `denom` is not array-like or None.

	ValueError
		If `num` and `denom` are both array-like but have different lengths.
	'''

	#check inputted data types and raise appropriate errors
	if not isinstance(num, (int, float, list, np.ndarray, type(None))):
		raise TypeError('num bust be scalar, array-like, or None')

	elif not isinstance(denom, (list, np.ndarray, type(None))):
		raise TypeError('denom bust be array-like, or None')

	#return 'undefined' if either is None
	if num is None or denom is None:
		return 'undefined'

	#take gradients, and clean-up if necessary
	try:
		dn = np.gradient(num)
	except ValueError:
		dn = np.zeros(len(denom))

	dd = np.gradient(denom)

	#check lengths and continuously increasing denom
	if len(dn) != len(dd):
		raise ValueError('num and denom arrays must have same length')

	elif any(dd) <= 0:
		return 'undefined'

	return np.array(dn/dd)





#define function to extract variables from 'real_data'
def _rpo_extract_tg(all_data, nT):
	'''
	Extracts time, temperature, and carbon remaining vectors from all_data.
	Called by ``RealData.__init__()``.

	Parameters
	----------
	all_data : str or pd.DataFrame 
		File containing thermogram data, either as a path string or 
		``pd.DataFrame`` instance.

	nT : int 
		The number of time points to use. Defaults to 250.

	Returns
	-------
	t : np.ndarray
		Array of timepoints (in seconds).

	Tau : np.ndarray
		Array of temperature points (in Kelvin).
	
	g : np.ndarray
		Array of fraction of carbon remaining.

	Raises
	------
	ValueError
		If `all_data` is not str or ``pd.DataFrame`` instance.
	
	ValueError
		If `all_data` does not contain "CO2_scaled" and "temp" columns.
	
	ValueError
		If index is not ``pd.DatetimeIndex`` instance.
	'''

	#import all_data as a pd.DataFrame if inputted as a string path and check
	#that it is in the right format
	if isinstance(all_data,str):
		all_data = pd.DataFrame.from_csv(all_data)
	elif not isinstance(all_data,pd.DataFrame):
		raise ValueError('all_data must be pd.DataFrame or path string')

	if 'CO2_scaled' and 'temp' not in all_data.columns:
		raise ValueError('all_data must have "CO2_scaled" and "temp" columns')

	if not isinstance(all_data.index,pd.DatetimeIndex):
		raise ValueError('all_data index must be DatetimeIndex')

	#extract necessary data
	secs = (all_data.index - all_data.index[0]).seconds
	CO2 = all_data.CO2_scaled
	alpha = np.cumsum(CO2)/np.sum(CO2)
	Temp = all_data.temp

	#generate t vector
	t0 = secs[0]; tf = secs[-1]; dt = (tf-t0)/nT
	t = np.linspace(t0,tf,nT+1) + dt/2 #make downsampled points at midpoint
	t = t[:-1] #drop last point since it's beyond tf

	#down-sample g and Tau using interp1d
	fT = interp1d(secs,Temp)
	fg = interp1d(secs,alpha)
	Tau = fT(t)
	g = 1-fg(t)
	
	return t, Tau, g