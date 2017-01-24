'''
Module to store all the core functions for ``rampedpyrox``.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['assert_len', 'calc_L_curve', 'derivatize', 'extract_moments']

import matplotlib.pyplot as plt
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
	Asserts that an array has length `n` and `float` datatypes.

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
	instance in order to choose the best-fit smoothing parameter, `omega`.

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
		Axis to plot on. If `None` and ``plot = True``, automatically 
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
		If ``plot = True``, returns an updated axis handle with plot.
	
	Raises
	------
	ScalarError
		If `om_max` or `om_min` are not int or float.

	ScalarError
		If `nOm` is not int.

	See Also
	--------
	Daem.calc_L_curve
		Instance method for ``calc_L_curve``.

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
		ax = ax, 
		plot = plot, 
		nOm = nOm, 
		om_max = om_max, 
		om_min = om_max)

#define function to derivatize an array w.r.t. another array
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
	numpy.gradient
		The method used to calculate derivatives

	Notes
	-----
	This method uses the ``np.gradient`` method to calculate derivatives. If
	`denom` is a scalar, resulting array will be all ``np.inf``. If both `num`
	and `denom` are scalars, resulting array will be all ``np.nan``. If 
	either `num` or `denom` are 1d and the other is 2d, derivative will be
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

#define function to extract 1st and 2nd moments for a distribution
def extract_moments(x, y):
	'''
	Extracts 1st (mean) and 2nd (stdev) moments from a distribution.

	Parameters
	----------
	x : np.ndarray
		Array of x values, length `n`.

	y : np.ndarray
		Array of y values, length `n`.

	Returns
	-------
	mu : float
		First moment of distribution.

	sigma : float
		Second moment of distribution.
	'''

	#assert lengths
	n = len(x)
	y = assert_len(y, n)

	#calculate first moment
	scalar = 1/np.sum(y*np.gradient(x))

	mu = np.sum(x*y*scalar*np.gradient(x))
	sigsq = np.sum((x - mu)**2 * y*scalar*np.gradient(x))
	sigma = sigsq**0.5

	return mu, sigma

#define function for plotting raw thermogram with isotopes
def plot_tg_isotopes(timedata, result, ax = None, plt_corr = True):
	'''
	Function to plot raw timedata (e.g. RPO thermogram) and isotope values.

	Parameters
	----------
	ax : None or matplotlib.axis
		Axis to plot on. If `None`, automatically creates a
		``matplotlip.axis`` instance to return. Defaults to `None`.
	
	plt_corr : str
		If `plt_var` is 'Fm' or 'd13C', `plt_corr` tells the method
		whether to plot raw or corrected values (if corrected values
		exist).

	result : rp.Results
		``rp.Results`` instance containing the isotope results to plot.

	timedata : rp.TimeData
		``rp.TimeData`` instance containing the derivative timedata (e.g. rpo
		 thermogram) to plot.

	Returns
	-------
	ax : matplotlib.axis
		Updated axis instance with plotted data.

	Warnings
	--------
	UserWarning
		If `timedata` does not contain derivative timedata, dgdt.
	
	UserWarning
		If `result` does not contain any of the necessary isotope attributes.


	ArrayError
		if `plt_corr` is `True` but no corrected data exist.

	ArrayError
		If `result` does not contain any of: d13C, Fm.
	'''

	#extract the timedata dgdt
	try:
		dgdt = timedata.dgdt

	except AttributeError:
		#raise warning
		warnings.warn(
			'TimeData instance does not contain dgdt attribute! Proceeding'
			' to plot isotopes only!')

		#proceed with dgdt = None
		dgdt = None

	#extract t array also
	t = timedata.t

	#extract isotope data
	if plt_corr is True:
		flag = '_corr'
	
	else:
		flag = '_raw'

	#check that isotope attributes exist and extract if so
	d13C = getattr(result, 'd13C' + flag, None)
	d13C_std = getattr(result, 'd13C' + flag + '_std', None)

	Fm = getattr(result, 'Fm' + flag, None)
	Fm_std = getattr(result, 'Fm' + flag + '_std', None)

	#raise warning if no isotope data exist
	if d13C is None and Fm is None:
		#raise warning
		warnings.warn(
			'Result instance does not contain d13C or Fm attributes!'
			' Proceeding to thermogram only!')

	#create axis if necessary
	if ax is None:
		_, ax = plt.subplots(1, 1)

	#calculate time elapsed in each fraction and midpoints
	DT = result.t_frac[:,1] - result.t_frac[:,0]
	x = result.t_frac[:,0] + DT/2

	#create string to add variables to for legend
	lns = []

	#plot d13C
	if d13C is not None:
		d13C_plot = ax.errorbar(
			x,
			d13C,
			yerr = d13C_std,
			marker = 'o',
			ecolor = 'k',
			markersize = 12,
			mec = 'k',
			mfc = 'w',
			elinewidth = 1,
			markeredgewidth = 1,
			capsize = 0,
			ls = 'none',
			zorder = 3,
			label = r'fraction $\delta^{13}C$')

		#set label
		ax.set_ylabel(r'$\delta^{13}C$ (VPDB)')

		#append legend
		lns.append(d13C_plot)

	#copy axis
	ax2 = ax.twinx()

	#plot Fm, if it exists
	if Fm is not None:

		Fm_plot = ax2.bar(
			result.t_frac[:,0],
			Fm,
			width = DT,
			color = 'none',
			linewidth = 1.5,
			zorder = 2,
			label = 'fraction Fm')

		#set label
		ax2.set_ylabel('Fm')

		#append legend
		lns.append(Fm_plot)

		#find max Fm value
		mFm = np.max(Fm)
	else:
		mFm = 1.0

	#add thermogram plot if it exists
	if dgdt is not None:
		
		#scale thermogram to match Fm
		mtg = np.max(-dgdt)
		tg_scaled = -dgdt*mFm/mtg

		#plot thermogram
		tg_plot = ax2.fill_between(
			t,
			np.zeros(len(t)),
			tg_scaled,
			facecolor = [0.5, 0.5, 0.5],
			edgecolor = 'k',
			alpha = 0.3,
			label = 'thermogram (normalized)',
			zorder = 1)

		#append legend
		lns.append(tg_plot)

	#set x label and limits
	ax.set_xlabel('time (seconds)')
	ax2.set_ylim([0, 1.1*mFm])

	#get legend of everything in one
	labs = [l.get_label() for l in lns]
	
	ax.legend(
		lns,
		labs, 
		loc = 'best',
		frameon = False)

	#make tight layout
	plt.tight_layout()

	return ax, ax2

