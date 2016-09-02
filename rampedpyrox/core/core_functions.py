'''
Module to store all the core functions for ``rampedpyrox``.
'''

import numpy as np

from collections import Sequence

from rampedpyrox.core.array_classes import(
	rparray
	)

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
	rampedpyrox.Daem.plot_L_curve
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

#define package-level function for derivatizing rparrays
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








