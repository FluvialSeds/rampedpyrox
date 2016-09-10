'''
Module to store all the core functions for ``rampedpyrox``.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['assert_len', 'calc_L_curve', 'derivatize']

import numpy as np

from collections import Sequence

#import exceptions
from .exceptions import(
	ArrayError,
	LengthError,
	)

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
	ArrayError
		If inputted data not int or array-like (excluding string).

	LengthError
		If length of the array is not n.
	'''


	#assert that n is int
	n = int(n)

	#assert data is in the right form
	if isinstance(data, (int, float)):
		data = data*np.ones(n)
	
	elif isinstance(data, Sequence) or hasattr(data, '__array__'):
		
		if isinstance(data, str):
			raise ArrayError(
				'Data cannot be a string')

		elif len(data) != n:
			raise LengthError(
				'Cannot create array of length %r if n = %r' \
				% (len(data), n))

	else:
		raise ArrayError('data must be scalar or array-like')

	return np.array(data).astype(float)

#define package-level function for calculating L curves
def calc_L_curve(
		model, 
		timedata, 
		ax = None, 
		plot = False, 
		nOm = 150, 
		om_max = 1e2, 
		om_min = 1e-3):
	'''
	Function to calculate the L-curve for a given model and timedata
	instance in order to choose the best-fit smoothing parameter, omega.

	Parameters
	----------
	model : rp.Model
		``rp.Model`` instance containing the A matrix to use for L curve 
		calculation.

	timedata : rp.TimeData
		``rp.TimeData`` instance containing the time and fraction remaining
		arrays to use in L curve calculation.

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
	
	Raises
	------
	ScalarError
		If `om_max` or `om_min` are not int or float.

	ScalarError
		If `nOm` is not int.

	See Also
	--------
	Daem.plot_L_curve
		Instance method for ``plot_L_curve``.

	References
	----------
	[1] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.

	[2] P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
		Numerical aspects of linear inversion (monographs on mathematical
		modeling and computation). *Society for Industrial and Applied
		Mathematics*.

	[3] P.C. Hansen (1994) Regularization tools: A Matlab package for analysis
		and solution of discrete ill-posed problems. *Numerical Algorithms*, 
		**6**, 1-35.
	'''

	return model.calc_L_curve(
		timedata, 
		ax=ax, 
		plot=plot, 
		nOm = 150, 
		om_max = 1e2, 
		om_min = 1e-3)

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
		The denominator of the numerical derivative function. Length `n`.

	Returns
	-------
	derivative : rparray
		An ``np.ndarray`` instance of the derivative. Length `n`.

	Raises
	------
	ArrayError
		If `denom` is not array-like.

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
				raise ArrayError('denom cannot be a string')

	else:
		raise ArrayError('denom must be array-like')

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
		col_der = [derivatize(col, denom) for col in num.T]
		dndd = np.column_stack(col_der)

	elif num.ndim == 1 and denom.ndim == 2:
		col_der = [derivatize(num, col) for col in denom.T]
		dndd = np.column_stack(col_der)

	return dndd
