'''
Module to store all the core functions for ``rampedpyrox``.
'''

import numpy as np

from collections import Sequence


#define function to assert length of array
def assert_len(data, n):
	'''
	Asserts that an array has length n. If data is scalar, creates an array of
	length n with repeating data as float.

	Parameters
	----------
	data : scalar or array-like
		Array to assert has length n. If scalar, generates an np.ndarray
		with length n.

	n : int
		Length to assert

	Returns
	-------
	array : np.ndarray
		Updated array, now of class np.ndarray and with length n.

	Raises
	------
	TypeError
		If inputted data not int or array-like (excluding string).

	ValueError
		If length of the array is not n.
	'''


	#assert that n is int
	n = int(n)

	#assert data is in the right form
	if isinstance(data, (int, float)):
		data = data*np.ones(n)
	
	elif isinstance(data, Sequence) or hasattr(data, '__array__'):
		
		if isinstance(data, str):
			raise TypeError('data cannot be a string')

		elif len(data) != n:
			raise ValueError(
				'Cannot create array of length %r if n = %r' \
				% (len(data), n))

	else:
		raise TypeError('data must be scalar or array-like')

	return np.array(data).astype(float)

#define package-level function for calculating L curves
def calc_L_curve(model, timedata, ax=None, plot=False, **kwargs):
	'''
	Function to calculate the L-curve for a given model and timedata
	instance in order to choose the best-fit smoothing parameter, omega.

	Parameters
	----------
	model : rp.Model
		Instance of ``Model`` subclass containing the A matrix to use for
		L curve calculation.

	timedata : rp.TimeData
		Instance of ``TimeData`` subclass containing the time and fraction
		remaining arrays to use in L curve calculation.

	Keyword Arguments
	-----------------
	ax : None or matplotlib.axis
		Axis to plot on. If `None` and ``plot=True``, automatically 
		creates a ``matplotlip.axis`` instance to return. Defaults to 
		`None`.

	plot : Boolean
		Tells the method to plot the resulting L curve or not.

	om_min : int
		Minimum omega value to search. Defaults to 1e-3.

	om_max : int
		Maximum omega value to search. Defaults to 1e2.

	nOm : int
		Number of omega values to consider. Defaults to 150.

	Returns
	-------
	om_best : float
		The calculated best-fit omega value.

	axis : None or matplotlib.axis
		If ``plot=True``, returns an updated axis handle with plot.
	
	Notes
	-----

	See Also
	--------
	Daem.plot_L_curve
		Instance method for ``plot_L_curve``.

	References
	----------
	D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
	respiration rates from decay time series. *Biogeosciences*, **9**,
	3601-3612.

	P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
	Numerical aspects of linear inversion (monographs on mathematical
	modeling and computation). *Society for Industrial and Applied
	Mathematics*.

	P.C. Hansen (1994) Regularization tools: A Matlab package for analysis and
	solution of discrete ill-posed problems. *Numerical Algorithms*, **6**,
	1-35.
	'''

	return model.calc_L_curve(timedata, ax=ax, plot=plot, **kwargs)

#define function to derivatize an array wrt another array
def derivatize(num, denom):
	'''
	Method for derivatizing numerator, `num`, with respect to denominator, 
	`denom`.

	Parameters
	----------
	num : int or array-like
		The numerator of the numerical derivative function.

	denom : array-like
		The denominator of the numerical derivative function. Length n.

	Returns
	-------
	derivative : rparray
		An ``np.ndarray`` instance of the derivative. Length n.

	Raises
	------
	TypeError
		If `denom` is not array-like.

	TypeError
		If `num` is not scalar or array-like.

	ValueError
		If `num` is not scalar or length n.

	See Also
	--------
	np.gradient
		The method used to calculate derivatives

	Notes
	-----
	This method uses the ``np.gradient`` method to calculate derivatives. If
	`denom` is a scalar, resulting array will be all ``np.inf``. If both `num`
	and `denom` are scalars, resulting array will be all ``np.nan``. If 
	either `num` or `self` are 1d and the other is 2d, derivative will be
	calculated column-wise. If both are 2d, each column will be derivatized 
	separately.
	'''

	#assert denom is the right type
	if isinstance(denom, Sequence) or hasattr(denom, '__array__'):
			if isinstance(denom, str):
				raise TypeError('denom cannot be a string')

	else:
		raise TypeError('denom must be array-like')

	#make sure the arrays are the same length, or convert num to array if 
	# scalar
	n = len(denom)
	num = assert_len(num, n)

	#calculate separately for each dimensionality case
	if num.ndim == denom.ndim == 1:
		dndd = np.gradient(num)/np.gradient(denom)

	elif num.ndim == denom.ndim == 2:
		dndd = np.gradient(num)[0]/np.gradient(denom)[0]

	#note recursive list comprehension when dimensions are different
	elif num.ndim == 2 and denom.ndim == 1:
		col_der = [_derivatize(col, denom) for col in num.T]
		dndd = np.column_stack(col_der)

	elif num.ndim == 1 and denom.ndim == 2:
		col_der = [_derivatize(num, col) for col in denom.T]
		dndd = np.column_stack(col_der)

	return dndd

#function to round to sig fig
def round_to_sigfig(vec, sig_figs=6):
	'''
	Rounds inputted vector to specified sig fig.

	Parameters
	----------
	vec : np.ndarray
		Array of data to round.

	sig_figs : int
		Number of sig figs to round to. Defaults to 6.

	Returns
	-------
	vec_round : np.ndarray
		`vec` array rounded to `sig_figs`.

	References
	----------
	D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
	respiration rates from decay time series. *Biogeosciences*, **9**,
	3601-3612.
	'''

	#create lambda function for rounding
	rnd = lambda x, n: round(x, -int(np.floor(np.log10(abs(x)))) + n - 1)

	#use list comprehension to round the vector
	vec_round = [rnd(x, sig_figs) for x in vec if x != 0]

	# p = sig_figs
	# order = np.floor(np.log10(vec))
	# vecH = 10**(p-order-1)*vec
	# vec_rnd_log = np.round(vecH)
	# vec_round = vec_rnd_log/10**(p-order-1)

	return vec_round





