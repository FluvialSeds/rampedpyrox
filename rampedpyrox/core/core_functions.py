'''
Module to store all the core functions for ``rampedpyrox``.
'''

import numpy as np

from collections import Sequence

from rampedpyrox.core.array_classes import(
	rparray
	)


def derivatize(num, denom, sig_figs=6):
	'''
	Method for derivatizing numerator, `num`, with respect to denominator, 
	`denom`.

	Parameters
	----------
	num : int or array-like
		The numerator of the numerical derivative function.

	denom : scalar or array-like
		The denominator of the numerical derivative function.

	Keyword Arguments
	-----------------
	sig_figs : int
		Number of significant figures to retain. Defaults to 6.
	
	Returns
	-------
	derivative : rparray
		An ``rparray`` instance of the derivative.

	See Also
	--------
	np.gradient
		The method used to calculate derivatives

	rparray.derivatize
		Instance method of ``derivatize``.

	Notes
	-----
	This method uses the ``np.gradient`` method to calculate derivatives.

	If `denom` is a scalar, resulting array will be all ``np.inf``.
	
	If both `num` and `denom` are scalars, resulting array will be all
	``np.nan``.

	If either `num` or `self` are 1d and the other is 2d, derivative
	will be calculated column-wise. If both are 2d, each column will
	be derivatized separately.

	'''

	#make sure num is the right length and type
	n = len(denom)
	num = rparray(num, n)

	return num.derivatize(denom, sig_figs=sig_figs)


